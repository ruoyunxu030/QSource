"""Market construction: sellers, segments, packets, trials, label noise."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from qsource.const import KIND_PACKET, KIND_TRIAL, KIND_TRIAL5

def _M_of_bool(F, M0, selected):
    M = np.asarray(M0, float).copy()
    for j, s in enumerate(selected):
        if s:
            M = M + F[j]
    return M


def _risk(Q, M, sigma2=1.0):
    return float(sigma2 * np.trace(Q @ np.linalg.inv(M)))


def greedy_plus_swap_cpu(F, c, M0, Q, B, sigma2=1.0, max_swap_rounds: int = 5):
    F = np.asarray(F, float)
    c = np.asarray(c, float)
    N = F.shape[0]
    selected = np.zeros(N, dtype=bool)
    spent = 0.0
    M = np.asarray(M0, float).copy()
    # greedy
    while True:
        try:
            loss = _risk(Q, M, sigma2)
        except np.linalg.LinAlgError:
            break
        best_j, best_score = -1, 0.0
        for j in range(N):
            if selected[j] or spent + c[j] > B + 1e-9:
                continue
            try:
                gain = loss - _risk(Q, M + F[j], sigma2)
            except np.linalg.LinAlgError:
                continue
            score = gain / c[j] if c[j] > 0 else gain
            if score > best_score:
                best_score, best_j = score, j
        if best_j < 0 or best_score <= 0:
            break
        selected[best_j] = True
        spent += c[best_j]
        M = M + F[best_j]
    # 1-swap with incremental M (avoid full rebuild)
    for _ in range(max_swap_rounds):
        try:
            loss = _risk(Q, M, sigma2)
        except np.linalg.LinAlgError:
            break
        spent = float(c[selected].sum()) if selected.any() else 0.0
        in_idx = np.where(selected)[0]
        out_idx = np.where(~selected)[0]
        best = None
        for i in in_idx:
            for j in out_idx:
                if spent - c[i] + c[j] > B + 1e-9:
                    continue
                try:
                    loss2 = _risk(Q, M - F[i] + F[j], sigma2)
                except np.linalg.LinAlgError:
                    continue
                gain = loss - loss2
                if gain > 1e-12 and (best is None or gain > best[0]):
                    best = (gain, i, j)
        if best is None:
            break
        _, i, j = best
        selected[i] = False
        selected[j] = True
        M = M - F[i] + F[j]
    return selected

N_SELLERS = 8
PKTS_PER_SELLER = 5
PKT_ROWS = 40
Q_LEVELS = [0.0, 0.1, 0.3, 0.5]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
S1_FIXED = 5

def noise_weight(q: float) -> float:
    """Effective Fisher / sample weight: (1−q)² signal attenuation."""
    return float(max(1.0 - q, 0.0) ** 2)


def inject_noise_amazon(y: np.ndarray, q: float, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(y, float).copy()
    if q <= 0:
        return y
    flip = rng.random(len(y)) < q
    y[flip] = rng.integers(1, 6, size=int(flip.sum())).astype(float)
    return y


def inject_noise_domainnet(
    y: np.ndarray, y_id: np.ndarray, q: float, rng: np.random.Generator
):
    y = np.asarray(y, float).copy()
    y_id = np.asarray(y_id).copy()
    C = y.shape[1]
    if q <= 0:
        return y, y_id
    flip = rng.random(len(y)) < q
    n_flip = int(flip.sum())
    if n_flip:
        new_id = rng.integers(0, C, size=n_flip)
        y_id[flip] = new_id
        y[flip] = 0.0
        y[flip, new_id] = 1.0
    return y, y_id


def draw_seed(rng, groups, used, n_seed, d, lam):
    blocks_X, blocks_y = [], []
    for g, dat in groups.items():
        Xs, ys = dat["X_sell"], dat["y_sell"]
        for i in range(len(Xs)):
            if i not in used.get(g, set()):
                blocks_X.append(Xs[i])
                blocks_y.append(ys[i])
    if len(blocks_X) < n_seed:
        blocks_X = [dat["X_sell"][i] for dat in groups.values() for i in range(len(dat["X_sell"]))]
        blocks_y = [dat["y_sell"][i] for dat in groups.values() for i in range(len(dat["y_sell"]))]
    idx = rng.choice(len(blocks_X), size=n_seed, replace=False)
    X0 = np.stack([blocks_X[i] for i in idx], axis=0)
    y0 = np.stack([blocks_y[i] for i in idx], axis=0)
    M0 = lam * np.eye(d) + X0.T @ X0
    return M0, X0, y0


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------


def select_open(F, c, M0, Sigma, B, sigma2):
    return greedy_plus_swap_cpu(F, c, M0, Sigma, B, sigma2=sigma2)


def select_extend(F, c, M0, Sigma, B, owned, sigma2):
    """Keep owned packets; fill remainder of budget with greedy+swap on rest."""
    owned = np.asarray(owned, bool)
    spent = float(c[owned].sum()) if owned.any() else 0.0
    rem = float(B) - spent
    if rem < 1e-9 or not (~owned).any():
        return owned.copy()
    M1 = _M_of_bool(F, M0, owned)
    remain = ~owned
    Fr, cr = F[remain], c[remain]
    loc = greedy_plus_swap_cpu(Fr, cr, M1, Sigma, rem, sigma2=sigma2)
    out = owned.copy()
    out[np.where(remain)[0][loc]] = True
    return out


def two_stage_proxy(F, c, M0, Sigma_pre, Sigmas, B, alpha, rho, sigma2):
    B1 = float(np.floor(alpha * B))
    s1 = select_open(F, c, M0, Sigma_pre, B1, sigma2)
    M1 = _M_of_bool(F, M0, s1)
    pre = _risk(Sigma_pre, M1, sigma2)
    post = 0.0
    for Sig in Sigmas:
        s2 = select_extend(F, c, M0, Sig, B, s1, sigma2)
        M2 = _M_of_bool(F, M0, s2)
        post += _risk(Sig, M2, sigma2)
    post /= len(Sigmas)
    return s1, pre + float(rho) * post


def best_alpha(F, c, M0, Sigma_pre, Sigmas, B, rho, sigma2, alphas=None):
    best_a, best_proxy = 0.5, np.inf
    for a in (alphas if alphas is not None else ALPHAS):
        _, proxy = two_stage_proxy(F, c, M0, Sigma_pre, Sigmas, B, a, rho, sigma2)
        if proxy < best_proxy:
            best_proxy, best_a = proxy, a
    return best_a

def mse_of(model, X, y):
    pred = model.predict(X)
    return float(np.mean((pred - y) ** 2))


def acc_of(model, X, y_id):
    if y_id is None:
        return float("nan")
    pred = model.predict(X)
    if pred.ndim == 1:
        return float("nan")
    return float(np.mean(np.argmax(pred, axis=1) == np.asarray(y_id)))


def gather_selected(sel, Xs, ys, yids, qs, pkt_weights=None, row_weights=None):
    """Gather selected packets.

    Priority: row_weights[j] (per-row array) > pkt_weights[j] (scalar) > noise_weight(qs[j]).
    """
    X_list, y_list, id_list, w_list = [], [], [], []
    for j, take in enumerate(sel):
        if not take:
            continue
        X_list.append(Xs[j])
        y_list.append(ys[j])
        n_j = len(Xs[j])
        if row_weights is not None and j in row_weights:
            wj = np.asarray(row_weights[j], float)
            if wj.ndim == 0:
                wj = np.full(n_j, float(wj))
            w_list.extend(wj.tolist())
        else:
            wj = float(pkt_weights[j]) if pkt_weights is not None else noise_weight(qs[j])
            w_list.extend([wj] * n_j)
        if yids[j] is not None:
            id_list.append(yids[j])
    if not X_list:
        return None, None, None, None
    X = np.vstack(X_list)
    y = np.concatenate(y_list, axis=0)
    yid = np.concatenate(id_list, axis=0) if id_list else None
    w = np.asarray(w_list, float)
    return X, y, yid, w


def fit_eval(
    sel,
    X0,
    y0,
    Xs,
    ys,
    yids,
    qs,
    lam,
    X_mix,
    y_mix,
    yid_mix,
    X_z,
    y_z,
    yid_z,
    also_weighted: bool = False,
    pkt_weights=None,
    primary_weighted: bool = False,
    row_weights=None,
):
    """Fit ridge on selected packets (+ seed).

    If primary_weighted and (row_weights or pkt_weights) given, the primary mse_*
    columns come from sample-weighted ridge.
    also_weighted still fills mse_*_w from weighted fit when requested.
    """
    Xp, yp, yidp, wp = gather_selected(
        sel, Xs, ys, yids, qs, pkt_weights=pkt_weights, row_weights=row_weights,
    )
    if Xp is None:
        X_all, y_all = X0, y0
        w_all = np.ones(len(X0))
        yid_all = None
    else:
        X_all = np.vstack([X0, Xp])
        y_all = np.concatenate([y0, yp], axis=0)
        w_all = np.concatenate([np.ones(len(X0)), wp], axis=0)
        yid_all = yidp

    use_w_primary = bool(
        primary_weighted and (row_weights is not None or pkt_weights is not None)
    )
    if use_w_primary:
        ridge = Ridge(alpha=lam, fit_intercept=False).fit(
            X_all, y_all, sample_weight=w_all
        )
    else:
        ridge = Ridge(alpha=lam, fit_intercept=False).fit(X_all, y_all)
    mse_mix = mse_of(ridge, X_mix, y_mix)
    mse_z = mse_of(ridge, X_z, y_z)
    acc_mix = acc_of(ridge, X_mix, yid_mix) if yid_mix is not None else np.nan
    acc_z = acc_of(ridge, X_z, yid_z) if yid_z is not None else np.nan
    out = dict(mse_mix=mse_mix, mse_z=mse_z, acc_mix=acc_mix, acc_z=acc_z)

    if also_weighted or use_w_primary:
        wr = Ridge(alpha=lam, fit_intercept=False).fit(X_all, y_all, sample_weight=w_all)
        out["mse_mix_w"] = mse_of(wr, X_mix, y_mix)
        out["mse_z_w"] = mse_of(wr, X_z, y_z)
        out["acc_mix_w"] = acc_of(wr, X_mix, yid_mix) if yid_mix is not None else np.nan
        out["acc_z_w"] = acc_of(wr, X_z, yid_z) if yid_z is not None else np.nan
    return out


def oracle_mse(groups, z_name, lam, X_test, y_test):
    model = Ridge(alpha=lam, fit_intercept=False).fit(
        groups[z_name]["X_sell"], groups[z_name]["y_sell"]
    )
    return mse_of(model, X_test, y_test)


def mixed_test(groups, demand):
    Xt = np.vstack([groups[g]["X_test"] for g in demand])
    yt = np.concatenate([groups[g]["y_test"] for g in demand], axis=0)
    if "y_id_test" in groups[demand[0]]:
        yid = np.concatenate([groups[g]["y_id_test"] for g in demand], axis=0)
    else:
        yid = None
    return Xt, yt, yid


def weighted_loss(pre, post, rho):
    return float(pre) + float(rho) * float(post)

def build_domainnet_confuse_map(groups) -> np.ndarray:
    """Nearest-other-class map from clean seller-pool class centroids (fixed once).

    Returns array ``confuse[c] = c'`` with c' ≠ c, shared by all instances/sellers.
    """
    Xs, yids = [], []
    for g, dat in groups.items():
        Xs.append(np.asarray(dat["X_sell"], float))
        yids.append(np.asarray(dat["y_id_sell"], int))
    X = np.vstack(Xs)
    yid = np.concatenate(yids)
    C = int(yid.max()) + 1
    means = np.zeros((C, X.shape[1]), float)
    for c in range(C):
        m = yid == c
        if not m.any():
            raise RuntimeError(f"domainnet confuse map: empty class {c}")
        means[c] = X[m].mean(axis=0)
    confuse = np.zeros(C, int)
    for c in range(C):
        d = np.linalg.norm(means - means[c], axis=1)
        d[c] = np.inf
        confuse[c] = int(np.argmin(d))
    return confuse


def _inject(dataset, Xsell, ysell, yid_sell, idx, q, rng,
            noise_type: str = "uniform", confuse_map=None):
    """Return (Xj, y_noisy, yid_noisy, y_clean, contam_mask).

    contam_mask[i]=True iff the injection Bernoulli flip fired for row i
    (may still equal clean label by chance after redraw).

    noise_type:
      uniform (default): existing Unif redraw — RNG stream unchanged.
      amazon_inflate / amazon_deflate: set flipped Amazon labels to 5 / 1.
      domainnet_confuse: map flipped DomainNet labels via confuse_map.
    """
    Xj = Xsell[idx]
    y_clean = np.asarray(ysell[idx], float).copy()
    n = len(idx)
    if q <= 0:
        contam = np.zeros(n, bool)
        if dataset == "amazon":
            return Xj, y_clean.copy(), None, y_clean, contam
        return Xj, y_clean.copy(), np.asarray(yid_sell[idx]).copy(), y_clean, contam
    contam = rng.random(n) < q
    if dataset == "amazon":
        yj = y_clean.copy()
        n_flip = int(contam.sum())
        if n_flip:
            if noise_type == "amazon_inflate":
                yj[contam] = 5.0
            elif noise_type == "amazon_deflate":
                yj[contam] = 1.0
            else:
                # uniform — must keep rng.integers for bit-identical markets
                yj[contam] = rng.integers(1, 6, size=n_flip).astype(float)
        return Xj, yj, None, y_clean, contam
    yj = y_clean.copy()
    yidj = np.asarray(yid_sell[idx]).copy()
    C = yj.shape[1]
    n_flip = int(contam.sum())
    if n_flip:
        if noise_type == "domainnet_confuse":
            if confuse_map is None:
                raise ValueError("domainnet_confuse requires confuse_map")
            old_id = yidj[contam].astype(int)
            new_id = np.asarray(confuse_map, int)[old_id]
            yidj[contam] = new_id
            yj[contam] = 0.0
            yj[contam, new_id] = 1.0
        else:
            new_id = rng.integers(0, C, size=n_flip)
            yidj[contam] = new_id
            yj[contam] = 0.0
            yj[contam, new_id] = 1.0
    return Xj, yj, yidj, y_clean, contam


def build_market_adaptive(rng, groups, dataset, s_rows: int,
                          noise_type: str = "uniform", confuse_map=None,
                          n_sellers: int | None = None,
                          add_entry_seller: bool = False,
                          entry_at: int = 15):
    """Packets (cost 1) + trial5 (cost 5/40) + trial (cost s/40), disjoint.

    Optional kwargs keep the default 8-seller market when left at defaults:
      n_sellers=None → N_SELLERS (8) per segment;
      add_entry_seller=False → no delayed seller.
    """
    names = list(groups.keys())
    Xs, ys, ys_clean, yids, contams = [], [], [], [], []
    src_row_idx = []
    sources, seller_ids, qs, costs = [], [], [], []
    kind_arr = []  # packet | trial5 | trial
    appear_at = []
    used = {g: set() for g in names}
    sid = 0
    c5 = float(S1_FIXED) / float(PKT_ROWS)
    cs = float(s_rows) / float(PKT_ROWS)
    if noise_type == "domainnet_confuse" and confuse_map is None:
        confuse_map = build_domainnet_confuse_map(groups)
    n_per = int(N_SELLERS if n_sellers is None else n_sellers)

    def emit_seller(g, q, appear_buyer):
        nonlocal sid
        Xsell, ysell = groups[g]["X_sell"], groups[g]["y_sell"]
        yid_sell = groups[g].get("y_id_sell")
        n = len(Xsell)
        seller_rows = []  # track source-row ids for this seller (disjointness)
        for _ in range(PKTS_PER_SELLER):
            avail = [i for i in range(n) if i not in used[g]]
            if len(avail) < PKT_ROWS:
                pool = avail if avail else list(range(n))
                idx = rng.choice(np.asarray(pool), size=PKT_ROWS, replace=True)
            else:
                idx = rng.choice(np.asarray(avail), size=PKT_ROWS, replace=False)
                used[g].update(idx.tolist())
            seller_rows.append(np.asarray(idx, int))
            Xj, yj, yidj, yc, cm = _inject(
                dataset, Xsell, ysell, yid_sell, idx, q, rng,
                noise_type=noise_type, confuse_map=confuse_map,
            )
            Xs.append(Xj); ys.append(yj); ys_clean.append(yc); yids.append(yidj)
            contams.append(cm); src_row_idx.append(np.asarray(idx, int))
            sources.append(g); seller_ids.append(sid); qs.append(float(q))
            costs.append(1.0); kind_arr.append(KIND_PACKET); appear_at.append(int(appear_buyer))
        pkt_now = set()
        for r in seller_rows[:PKTS_PER_SELLER]:
            pkt_now.update(int(x) for x in r.tolist())

        def _draw_trial_idx(k):
            # The default 8-seller markets never exhaust the unused pool. Larger markets can,
            # so packet and trial rows of a seller are kept disjoint here.
            avail = [i for i in range(n) if i not in used[g] and i not in pkt_now]
            if len(avail) >= k:
                idx = rng.choice(np.asarray(avail), size=k, replace=False)
                used[g].update(idx.tolist())
                return np.asarray(idx, int)
            disjoint = [i for i in range(n) if i not in pkt_now]
            pool = avail if avail else (disjoint if disjoint else list(range(n)))
            return rng.choice(np.asarray(pool), size=k, replace=True)

        # discarded 5-row trial (RNG must still run)
        idx = _draw_trial_idx(S1_FIXED)
        seller_rows.append(np.asarray(idx, int))
        Xj, yj, yidj, yc, cm = _inject(
            dataset, Xsell, ysell, yid_sell, idx, q, rng,
            noise_type=noise_type, confuse_map=confuse_map,
        )
        Xs.append(Xj); ys.append(yj); ys_clean.append(yc); yids.append(yidj)
        contams.append(cm); src_row_idx.append(np.asarray(idx, int))
        sources.append(g); seller_ids.append(sid); qs.append(float(q))
        costs.append(c5); kind_arr.append(KIND_TRIAL5); appear_at.append(int(appear_buyer))
        idx = _draw_trial_idx(s_rows)
        seller_rows.append(np.asarray(idx, int))
        Xj, yj, yidj, yc, cm = _inject(
            dataset, Xsell, ysell, yid_sell, idx, q, rng,
            noise_type=noise_type, confuse_map=confuse_map,
        )
        Xs.append(Xj); ys.append(yj); ys_clean.append(yc); yids.append(yidj)
        contams.append(cm); src_row_idx.append(np.asarray(idx, int))
        sources.append(g); seller_ids.append(sid); qs.append(float(q))
        costs.append(cs); kind_arr.append(KIND_TRIAL); appear_at.append(int(appear_buyer))
        pkt_set = set()
        for r in seller_rows[:PKTS_PER_SELLER]:
            pkt_set.update(int(x) for x in r.tolist())
        trial_set = set(int(x) for x in seller_rows[PKTS_PER_SELLER].tolist()) | set(
            int(x) for x in seller_rows[PKTS_PER_SELLER + 1].tolist()
        )
        inter = pkt_set & trial_set
        if inter:
            raise RuntimeError(
                f"row overlap within seller {sid} source={g}: "
                f"{len(inter)} rows shared between packets and purchased samples"
            )
        sid += 1

    for g in names:
        q_assign = list(Q_LEVELS) * (n_per // len(Q_LEVELS))
        rng.shuffle(q_assign)
        for q in q_assign:
            emit_seller(g, q, 1)

    entry_seller_id = None
    if add_entry_seller:
        entry_seller_id = int(sid)
        emit_seller(names[0], 0.0, int(entry_at))

    F = packet_fishers(Xs)
    w = np.array([noise_weight(q) for q in qs], float)
    kind_arr = np.asarray(kind_arr)
    is_trial = kind_arr != KIND_PACKET
    return dict(
        F=F, F_qual=F * w[:, None, None], c=np.asarray(costs, float),
        Xs=Xs, ys=ys, ys_clean=ys_clean, contams=contams, yids=yids,
        src_row_idx=src_row_idx,
        sources=np.asarray(sources),
        seller_ids=np.asarray(seller_ids, int), qs=np.asarray(qs, float),
        kind=kind_arr, is_trial=is_trial, used=used, names=names,
        s_rows=s_rows, trial_cost_s=cs, trial_cost_5=c5,
        noise_type=str(noise_type),
        appear_at=np.asarray(appear_at, int),
        offer_id=np.arange(len(costs), dtype=int),
        entry_seller_id=entry_seller_id,
        n_sellers_per_segment=n_per,
    )


def append_floor_trials(
    market: dict,
    rng,
    groups,
    dataset: str,
    s_rows: int,
    n_extra: int,
    noise_type: str = "uniform",
    confuse_map=None,
):
    """Add extra trial offers per seller for the evidence floor. Separate RNG.

    Prefers unused leftover rows. If leftover < s, sample with replacement
    from this seller's non-packet rows (I1 rule). Default n_extra=0 is unused;
    call only when floor inventory is needed. Does not replay market RNG.
    """
    n_extra = max(int(n_extra), 0)
    if n_extra <= 0:
        market.setdefault("floor_offer_ids", {})
        market.setdefault("floor_reuse_n", 0)
        return market
    s_rows = int(s_rows)
    used = market.get("used") or {}
    Xs = list(market["Xs"])
    ys = list(market["ys"])
    ys_clean = list(market["ys_clean"])
    yids = list(market["yids"])
    contams = list(market["contams"])
    src_row_idx = list(market.get("src_row_idx") or [None] * len(Xs))
    sources = list(np.asarray(market["sources"]))
    seller_ids = list(np.asarray(market["seller_ids"], int))
    qs = list(np.asarray(market["qs"], float))
    costs = list(np.asarray(market["c"], float))
    kind_arr = list(np.asarray(market["kind"]))
    appear = list(np.asarray(market["appear_at"], int))
    offer_id = list(np.asarray(market["offer_id"], int))
    oid0 = int(max(offer_id)) + 1 if offer_id else 0
    cs = float(s_rows) / float(PKT_ROWS)
    first_trial = {}
    extras: dict[int, list[int]] = {}
    pkt_src: dict[int, set[int]] = {}
    appear_s: dict[int, int] = {}
    src_s: dict[int, str] = {}
    q_s: dict[int, float] = {}
    for i, (s, k) in enumerate(zip(seller_ids, kind_arr)):
        s = int(s)
        appear_s[s] = int(appear[i])
        src_s[s] = str(sources[i])
        q_s[s] = float(qs[i])
        if k == KIND_PACKET and src_row_idx[i] is not None:
            pkt_src.setdefault(s, set()).update(int(x) for x in np.asarray(src_row_idx[i]).tolist())
        if k != KIND_TRIAL:
            continue
        if s not in first_trial:
            first_trial[s] = int(offer_id[i])
        else:
            extras.setdefault(s, []).append(int(offer_id[i]))
    reuse = int(market.get("floor_reuse_n") or 0)
    for s in sorted(appear_s):
        have = list(extras.get(s, []))
        need = n_extra - len(have)
        if need <= 0:
            extras[s] = have
            continue
        g = src_s[s]
        if g not in used:
            used[g] = set()
        Xsell, ysell = groups[g]["X_sell"], groups[g]["y_sell"]
        yid_sell = groups[g].get("y_id_sell")
        n = len(Xsell)
        pkt = pkt_src.get(s, set())
        q = q_s[s]
        for _ in range(need):
            avail = [i for i in range(n) if i not in used[g] and i not in pkt]
            if len(avail) >= s_rows:
                idx = rng.choice(np.asarray(avail), size=s_rows, replace=False)
                used[g].update(int(x) for x in idx.tolist())
            else:
                pool = avail if avail else [i for i in range(n) if i not in pkt]
                if not pool:
                    pool = list(range(n))
                idx = rng.choice(np.asarray(pool), size=s_rows, replace=True)
                reuse += 1
            Xj, yj, yidj, yc, cm = _inject(
                dataset, Xsell, ysell, yid_sell, idx, q, rng,
                noise_type=noise_type, confuse_map=confuse_map,
            )
            Xs.append(Xj); ys.append(yj); ys_clean.append(yc); yids.append(yidj)
            contams.append(cm); src_row_idx.append(np.asarray(idx, int))
            sources.append(g); seller_ids.append(s); qs.append(q)
            costs.append(cs); kind_arr.append(KIND_TRIAL); appear.append(appear_s[s])
            have.append(int(oid0))
            offer_id.append(oid0); oid0 += 1
        extras[s] = have
    F = packet_fishers(Xs)
    w = np.array([noise_weight(q_) for q_ in qs], float)
    kind_arr = np.asarray(kind_arr)
    out = dict(market)
    out.update(
        F=F, F_qual=F * w[:, None, None], c=np.asarray(costs, float),
        Xs=Xs, ys=ys, ys_clean=ys_clean, contams=contams, yids=yids,
        src_row_idx=src_row_idx, used=used,
        sources=np.asarray(sources),
        seller_ids=np.asarray(seller_ids, int), qs=np.asarray(qs, float),
        kind=kind_arr, is_trial=kind_arr != KIND_PACKET,
        appear_at=np.asarray(appear, int),
        offer_id=np.asarray(offer_id, int),
        floor_offer_ids={int(s): [int(x) for x in extras.get(s, [])] for s in appear_s},
        floor_reuse_n=int(reuse),
    )
    return out

def idx_of(seller_ids, kind_arr, s, kind):
    for j in range(len(seller_ids)):
        if seller_ids[j] == s and kind_arr[j] == kind:
            return j
    return -1


def rank_sellers(F, M0, Sigma, sigma2, seller_ids, kind_arr):
    pk = kind_arr == KIND_PACKET
    try:
        loss0 = _risk(Sigma, M0, sigma2)
    except np.linalg.LinAlgError:
        loss0 = 1e9
    best = {}
    for j in np.where(pk)[0]:
        s = int(seller_ids[j])
        try:
            gain = loss0 - _risk(Sigma, M0 + F[j], sigma2)
        except np.linalg.LinAlgError:
            gain = 0.0
        if s not in best or gain > best[s]:
            best[s] = gain
    return sorted(best.items(), key=lambda kv: -kv[1])

def strategy_q2(F, Fq, c, M0, Sigma_pre, Sigmas, B, rho, sigma2, kind_arr, z_idx):
    pk = kind_arr == KIND_PACKET
    Fqp, cp = Fq[pk], c[pk]
    alpha = best_alpha(Fqp, cp, M0, Sigma_pre, Sigmas, B, rho, sigma2)
    B1 = float(np.floor(alpha * B))
    loc1 = select_open(Fqp, cp, M0, Sigma_pre, B1, sigma2)
    owned = np.zeros(len(cp), bool); owned[loc1] = True
    loc2 = select_extend(Fqp, cp, M0, Sigmas[z_idx], B, owned, sigma2)
    out = np.zeros(len(c), bool)
    out[np.where(pk)[0][loc2]] = True
    return out, alpha

def build_market(rng, groups, dataset, s_rows: int,
                 noise_type: str = "uniform", confuse_map=None,
                 n_sellers: int | None = None,
                 add_entry_seller: bool = False,
                 entry_at: int = 15):
    """Packets + trial offers. The 5-row trial is generated then dropped so the RNG matches."""
    m = build_market_adaptive(
        rng, groups, dataset, s_rows,
        noise_type=noise_type, confuse_map=confuse_map,
        n_sellers=n_sellers, add_entry_seller=add_entry_seller,
        entry_at=entry_at,
    )
    keep = m["kind"] != KIND_TRIAL5
    out = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v)
           for k, v in m.items()}
    for k in ("Xs", "ys", "ys_clean", "contams", "yids", "src_row_idx"):
        if k in m:
            out[k] = [m[k][j] for j in np.where(keep)[0]]
    out["kind"] = m["kind"][keep]
    out["is_trial"] = out["kind"] != KIND_PACKET
    out["F"] = m["F"][keep]
    out["F_qual"] = m["F_qual"][keep]
    out["c"] = m["c"][keep]
    out["seller_ids"] = m["seller_ids"][keep]
    out["sources"] = m["sources"][keep]
    out["qs"] = m["qs"][keep]
    out["s_rows"] = s_rows
    out["trial_cost_s"] = float(s_rows) / float(PKT_ROWS)
    return out


def packet_fishers(Xs):
    Fs = []
    d = None
    for Xj in Xs:
        Xj = np.asarray(Xj, float)
        if Xj.ndim == 1:
            Xj = Xj.reshape(1, -1)
        if d is None:
            d = Xj.shape[1]
        elif Xj.shape[1] != d:
            raise ValueError(f"inconsistent d: {Xj.shape[1]} vs {d}")
        Fs.append(Xj.T @ Xj)
    return np.stack(Fs, axis=0)


def second_moment(X):
    return (X.T @ X) / max(len(X), 1)


def task_kind(dataset: str) -> str:
    return "regression" if dataset == "amazon" else "classification"


def n_extra_trials(s_rows: int) -> int:
    import math
    return int(math.ceil(80 / float(s_rows)))


def load_dataset(name: str, data_dir):
    """Load processed Amazon / DomainNet groups from data_dir."""
    from pathlib import Path
    import json
    data_dir = Path(data_dir)
    name = str(name)
    if name == "amazon":
        gdir = data_dir / "amazon"
        gmeta = json.loads((gdir / "amazon_meta.json").read_text())
        dmeta = json.loads((gdir / "dataset_meta.json").read_text())
        enc = dmeta.get("encoder") or gmeta.get("primary_encoder") or "bge-base-en-v1.5"
        groups = {}
        for cat in gmeta["demand_categories"]:
            z = np.load(gdir / f"cat_{enc}_{cat}.npz", allow_pickle=True)
            groups[cat] = {k: z[k] for k in z.files if k != "category"}
        return dict(groups=groups, sigma2=float(dmeta["sigma2"]),
                    lam=float(dmeta["ridge_lambda"]), d=int(dmeta["d"]))
    if name == "domainnet":
        gdir = data_dir / "domainnet"
        enc = "clip-vit-l14"
        gmeta = json.loads((gdir / f"domainnet_meta_{enc}.json").read_text())
        dmeta = json.loads((gdir / "dataset_meta.json").read_text())
        groups = {}
        for cat in gmeta["demand_categories"]:
            z = np.load(gdir / f"cat_{enc}_{cat}.npz")
            groups[cat] = dict(
                X_sell=z["X_sell"], y_sell=z["y_sell"],
                X_query=z["X_query"], X_test=z["X_test"], y_test=z["y_test"],
            )
            if "y_id_sell" in z.files:
                groups[cat]["y_id_sell"] = z["y_id_sell"]
                groups[cat]["y_id_test"] = z["y_id_test"]
        return dict(groups=groups, sigma2=float(dmeta["sigma2"]),
                    lam=float(dmeta["ridge_lambda"]), d=int(dmeta["d"]))
    raise ValueError(f"unknown dataset {name}")

