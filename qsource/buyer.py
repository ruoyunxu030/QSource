"""One buyer: draw a task, run trials, buy packets, write scores."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from qsource.const import BETA_A, BETA_B, KIND_PACKET, PKT_ROWS
from qsource.market import (
    _risk,
    best_alpha,
    draw_seed,
    fit_eval,
    mixed_test,
    oracle_mse,
    second_moment,
    strategy_q2,
    task_kind,
    weighted_loss,
)
from qsource.acquisition import _M_weighted, acquire
from qsource.store import BuyerEvidenceStore, EvidenceStore


def slice_market(market: dict, keep: np.ndarray) -> dict:
    keep = np.asarray(keep, bool)
    n = len(market["c"])
    out = {}
    for k, v in market.items():
        if isinstance(v, np.ndarray) and len(v) == n:
            out[k] = v[keep]
        elif isinstance(v, list) and len(v) == n:
            out[k] = [v[j] for j in np.where(keep)[0]]
        else:
            out[k] = v
    out["is_trial"] = out["kind"] != KIND_PACKET
    return out


def visible_market(market: dict, buyer_t: int) -> dict:
    appear = market.get("appear_at")
    if appear is None:
        return market
    keep = np.asarray(appear, int) <= int(buyer_t)
    if bool(np.all(keep)):
        return market
    return slice_market(market, keep)


def draw_buyer_task(buyer_rng, groups, market, n_seed, d, lam, K):
    names = list(groups.keys())
    demand = list(buyer_rng.choice(names, size=min(K, len(names)), replace=False))
    M0, X0, y0 = draw_seed(buyer_rng, groups, market["used"], n_seed, d, lam)
    Sigmas = [second_moment(groups[g]["X_query"]) for g in demand]
    Sigma_pre = sum(Sigmas) / len(Sigmas)
    X_mix, y_mix, yid_mix = mixed_test(groups, demand)
    return demand, Sigmas, Sigma_pre, M0, X0, y0, X_mix, y_mix, yid_mix


def _row_array(row_w, j, n):
    if row_w is None:
        return np.ones(n, float)
    w = row_w.get(j, row_w.get(int(j)))
    if w is None:
        return np.ones(n, float)
    w = np.asarray(w, float)
    if w.ndim == 0:
        return np.full(n, float(w))
    return w


def _naive_share_caps(store, vis_sids, s_rows, B, beta, trial_cost):
    vis_sids = [int(s) for s in vis_sids]
    n_vis = max(len(vis_sids), 1)
    n_distinct = {s: store.n_distinct(s) for s in vis_sids}
    n_insuf = sum(1 for s in vis_sids if n_distinct[s] < int(s_rows))
    if n_insuf == 0:
        n_trials = 0
    else:
        n_trials = max(
            1,
            int(np.floor(beta * B * n_insuf / n_vis / max(trial_cost, 1e-9))),
        )
    skip = {s for s in vis_sids if n_distinct[s] >= int(s_rows)}
    return n_trials, skip, n_distinct


def _cal_share_caps(bstore: BuyerEvidenceStore, vis_sids, B, beta, trial_cost,
                    skip_threshold: float = 1.0):
    vis_sids = [int(s) for s in vis_sids]
    n_vis = max(len(vis_sids), 1)
    score = getattr(bstore, "skip_score", None)
    n_map = {
        s: float(score(s) if score is not None else bstore.n_buyers(s))
        for s in vis_sids
    }
    thr = float(skip_threshold)
    n_insuf = sum(1 for s in vis_sids if n_map[s] < thr - 1e-12)
    if n_insuf == 0:
        n_trials = 0
    else:
        n_trials = max(
            1,
            int(np.floor(beta * B * n_insuf / n_vis / max(trial_cost, 1e-9))),
        )
    skip = {s for s in vis_sids if n_map[s] >= thr - 1e-12}
    return n_trials, skip


def _pi_from_store(store, vis_sids, policy, rng):
    pi_override = {}
    for s in vis_sids:
        a, b, mean, n = store.seller_posterior(s)
        if n <= 0:
            continue
        if policy == "share_ts":
            pi_override[s] = float(np.clip(rng.beta(a, b), 1e-3, 1.0 - 1e-3))
        else:
            pi_override[s] = float(np.clip(mean, 1e-3, 1.0 - 1e-3))
    return pi_override


def means_from_buyer_store(bstore: BuyerEvidenceStore, vis_sids) -> dict[int, float]:
    out = {}
    for s in vis_sids:
        m, _v, n, *_ = bstore.publish(int(s))
        # n is publish()'s third value. PurchaseWeightStore returns
        # round(n_w) if n_w≥1 else 0, so this gate is a decision.
        if n <= 0:
            continue
        out[int(s)] = float(np.clip(m, 0.05, 1.0))
    return out


def _share_kwargs(policy, store, vis_sids, s_rows, B, beta, trial_cost, rng,
                  buyer_store=None, skip_threshold: float = 1.0):
    if policy == "no_sharing" or (store is None and buyer_store is None):
        return {}
    vis_sids = [int(s) for s in vis_sids]
    if policy in ("buyer_level", "buyer_level_floor") and buyer_store is not None:
        n_trials, skip = _cal_share_caps(
            buyer_store, vis_sids, B, beta, trial_cost,
            skip_threshold=skip_threshold,
        )
        pi_override = means_from_buyer_store(buyer_store, vis_sids)
        return dict(
            n_trials_override=n_trials,
            skip_trial_sellers=skip,
            pi_override=pi_override,
            extra_trial_sellers=None,
            extra_trial_budget=0.0,
            allow_sellers=set(vis_sids),
        )
    if store is None:
        return {}
    n_trials, skip, _n_distinct = _naive_share_caps(
        store, vis_sids, s_rows, B, beta, trial_cost,
    )
    pi_override = _pi_from_store(store, vis_sids, policy, rng)
    return dict(
        n_trials_override=n_trials,
        skip_trial_sellers=skip,
        pi_override=pi_override,
        extra_trial_sellers=None,
        extra_trial_budget=0.0,
        allow_sellers=set(vis_sids),
    )


def _payments(sel, c, seller_ids, kind_arr, Xs, qs):
    sel = np.asarray(sel, bool)
    rows = []
    for s in np.unique(seller_ids):
        m = sel & (seller_ids == s)
        if not m.any():
            continue
        pkt = m & (kind_arr == KIND_PACKET)
        n_rows = int(sum(len(Xs[j]) for j in np.where(m)[0]))
        q = float(qs[m][0])
        rows.append(dict(
            seller=int(s),
            true_pi=float(1.0 - q),
            paid=float(c[m].sum()),
            packets_bought=int(pkt.sum()),
            rows_bought=n_rows,
            packet_paid=float(c[pkt].sum()) if pkt.any() else 0.0,
        ))
    return rows


def buyer_pi_hats(gamma_by_row, vis) -> dict[int, tuple[float, int]]:
    offer_id = np.asarray(vis["offer_id"], int)
    seller_ids = np.asarray(vis["seller_ids"], int)
    loc = {int(oid): i for i, oid in enumerate(offer_id)}
    by_s: dict[int, list[float]] = {}
    for rid, g in gamma_by_row.items():
        oid = int(rid[0])
        j = loc.get(oid)
        if j is None:
            continue
        s = int(seller_ids[j])
        by_s.setdefault(s, []).append(float(g))
    return {s: (float(np.mean(vs)), int(len(vs))) for s, vs in by_s.items()}


def run_one_buyer(
    policy: str,
    vis: dict,
    groups,
    demand,
    Sigmas,
    Sigma_pre,
    M0, X0, y0,
    X_mix, y_mix, yid_mix,
    cfg: dict,
    rng,
    store: EvidenceStore | None,
    buyer_store: BuyerEvidenceStore | None = None,
    exhaust_budget: bool = False,
    platform_trial_idx=None,
    extra_skip_sellers=None,
    qs_skip=None,
    qs_n_trials=None,
    qs_pi=None,
    trial_trace=None,
    skip_threshold: float = 1.0,
):
    dataset = cfg["dataset"]
    beta = float(cfg["beta"])
    s_rows = int(cfg["s_rows"])
    B = float(cfg["B"])
    rho = float(cfg["rho"])
    lam, sigma2 = cfg["lam"], cfg["sigma2"]
    kind = task_kind(dataset)
    F, c = vis["F"], vis["c"]
    Xs, ys, yids = vis["Xs"], vis["ys"], vis["yids"]
    ys_clean = vis.get("ys_clean")
    qs, seller_ids = vis["qs"], vis["seller_ids"]
    sources, kind_arr = vis["sources"], vis["kind"]
    offer_id = vis["offer_id"]
    trial_cost = float(s_rows) / float(PKT_ROWS)
    vis_sids = sorted({int(s) for s in seller_ids})
    random_trial = str(cfg.get("sampling_rule", "value")) == "uniform"

    excesses, losses, trial_spends, packet_spends = [], [], [], []
    pay_sum = {}
    gamma_lists: dict[tuple[int, int], list[float]] = {}
    pkt_sellers = set()
    n_checked_acc = []

    for zi, z_name in enumerate(demand):
        Xz = groups[z_name]["X_test"]
        yz = groups[z_name]["y_test"]
        yidz = groups[z_name].get("y_id_test")
        mse_star = oracle_mse(groups, z_name, lam, Xz, yz)
        mz = Ridge(alpha=lam, fit_intercept=False).fit(
            groups[z_name]["X_sell"], groups[z_name]["y_sell"],
        )
        mse_star_mix = float(np.mean((mz.predict(X_mix) - y_mix) ** 2))
        L_star = weighted_loss(mse_star_mix, mse_star, rho)
        buy_kw = _share_kwargs(
            policy, store, vis_sids, s_rows, B, beta, trial_cost, rng,
            buyer_store=buyer_store, skip_threshold=skip_threshold,
        )
        if exhaust_budget:
            buy_kw["exhaust_budget"] = True
        if trial_trace is not None:
            buy_kw["trial_trace"] = trial_trace
        if platform_trial_idx:
            buy_kw["platform_trial_idx"] = list(platform_trial_idx)
        if extra_skip_sellers:
            skip = set(buy_kw.get("skip_trial_sellers") or [])
            skip |= {int(s) for s in extra_skip_sellers}
            buy_kw["skip_trial_sellers"] = skip
        if qs_pi is not None:
            buy_kw["pi_override"] = {int(s): float(p) for s, p in qs_pi.items()}
            buy_kw["allow_sellers"] = set(vis_sids)
            skip = set(buy_kw.get("skip_trial_sellers") or [])
            skip |= {int(s) for s in (qs_skip or [])}
            buy_kw["skip_trial_sellers"] = skip
            if qs_n_trials is not None:
                buy_kw["n_trials_override"] = max(0, int(qs_n_trials))

        row_w = None
        sw = {}
        psp = 0.0
        nchk = 0
        if policy == "known_quality":
            sel, alph = strategy_q2(
                F, vis["F_qual"], c, M0, Sigma_pre, Sigmas, B, rho, sigma2, kind_arr, zi,
            )
            primary_w = False
        else:
            sel, alph, _tr, _qe, _qt, nchk, psp, _pr, sw, row_w = acquire(
                F, c, M0, Sigma_pre, Sigmas, B, rho, sigma2, beta, s_rows,
                seller_ids, sources, Xs, ys, qs, X0, y0, lam, kind, kind_arr,
                zi, rng, reuse=True, use_quality=True, random_trial=random_trial,
                ys_clean=ys_clean, yids=yids, **buy_kw,
            )
            primary_w = True

        pkt_w = None
        if primary_w and sw:
            pkt_w = np.array([float(sw.get(int(s), 1.0)) for s in seller_ids], float)
        ev = fit_eval(
            sel, X0, y0, Xs, ys, yids, qs, lam,
            X_mix, y_mix, yid_mix, Xz, yz, yidz,
            also_weighted=False, pkt_weights=pkt_w, primary_weighted=primary_w,
            row_weights=row_w,
        )
        loss = weighted_loss(ev["mse_mix"], ev["mse_z"], rho)
        losses.append(loss)
        excesses.append(loss - L_star)
        trial_spends.append(float(psp))
        packet_spends.append(float(c[sel & (kind_arr == KIND_PACKET)].sum()) if sel.any() else 0.0)
        n_checked_acc.append(int(nchk))

        buyer_sel = np.asarray(sel, bool).copy()
        if platform_trial_idx:
            for pj in platform_trial_idx:
                if 0 <= int(pj) < len(buyer_sel):
                    buyer_sel[int(pj)] = False
        for rec in _payments(buyer_sel, c, seller_ids, kind_arr, Xs, qs):
            s = rec["seller"]
            acc = pay_sum.setdefault(s, dict(
                true_pi=rec["true_pi"], paid=0.0, packets_bought=0.0,
                rows_bought=0.0, packet_paid=0.0, n=0,
            ))
            acc["paid"] += rec["paid"]
            acc["packets_bought"] += rec["packets_bought"]
            acc["rows_bought"] += rec["rows_bought"]
            acc["packet_paid"] += rec["packet_paid"]
            if rec["packets_bought"] > 0:
                pkt_sellers.add(s)

        if row_w is not None:
            for j in np.where(np.asarray(sel, bool))[0]:
                oid = int(offer_id[j])
                w = _row_array(row_w, j, len(Xs[j]))
                for i, gi in enumerate(w):
                    gamma_lists.setdefault((oid, i), []).append(float(gi))

    n_z = max(len(demand), 1)
    plat_paid_by_s: dict[int, float] = {}
    plat_rows_by_s: dict[int, int] = {}
    if platform_trial_idx:
        for pj in platform_trial_idx:
            pj = int(pj)
            if 0 <= pj < len(c):
                s = int(seller_ids[pj])
                plat_paid_by_s[s] = plat_paid_by_s.get(s, 0.0) + float(c[pj])
                plat_rows_by_s[s] = plat_rows_by_s.get(s, 0) + int(len(Xs[pj]))
    plat_paid = float(sum(plat_paid_by_s.values()))
    plat_rows = int(sum(plat_rows_by_s.values()))
    plat_seller = next(iter(plat_paid_by_s)) if len(plat_paid_by_s) == 1 else None
    payments = []
    seen_pay = set()
    for s, acc in pay_sum.items():
        seen_pay.add(int(s))
        payments.append(dict(
            seller=int(s),
            true_pi=float(acc["true_pi"]),
            paid=float(acc["paid"] / n_z),
            packets_bought=float(acc["packets_bought"] / n_z),
            rows_bought=float(acc["rows_bought"] / n_z),
            packet_paid=float(acc["packet_paid"] / n_z),
            bought_packet=int(s in pkt_sellers),
            platform_paid=float(plat_paid_by_s.get(int(s), 0.0)),
        ))
    for s, paid in plat_paid_by_s.items():
        if int(s) in seen_pay:
            continue
        qv = float(qs[seller_ids == s][0]) if np.any(seller_ids == s) else np.nan
        payments.append(dict(
            seller=int(s), true_pi=float(1.0 - qv),
            paid=0.0, packets_bought=0.0, rows_bought=0.0, packet_paid=0.0,
            bought_packet=0, platform_paid=float(paid),
        ))
    gamma_by_row = {k: float(np.mean(v)) for k, v in gamma_lists.items()}
    pkt_spend = float(np.mean(packet_spends)) if packet_spends else 0.0
    pr_spend = float(np.mean(trial_spends)) if trial_spends else 0.0
    summary = dict(
        excess=float(np.mean(excesses)),
        test_loss=float(np.mean(losses)) if losses else np.nan,
        trial_spend=pr_spend,
        packet_spend=pkt_spend,
        n_checked=float(np.mean(n_checked_acc)) if n_checked_acc else 0.0,
        n_z=n_z,
        platform_paid=float(plat_paid),
        platform_seller=plat_seller,
        platform_rows=int(plat_rows),
        platform_by_seller={
            int(s): dict(paid=float(plat_paid_by_s[s]), rows=int(plat_rows_by_s.get(s, 0)))
            for s in plat_paid_by_s
        },
    )
    return summary, payments, gamma_by_row, pkt_sellers
