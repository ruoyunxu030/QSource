"""Buyer purchase rule: trial ranking, contamination weights, packet fill."""
from __future__ import annotations

import numpy as np

from qsource.const import KIND_PACKET, KIND_TRIAL
from qsource.market import (
    PKT_ROWS,
    _risk,
    best_alpha,
    idx_of,
    rank_sellers,
)


def _M_weighted(F, M0, selected, w_pkt):
    """M = M0 + Σ_j selected w_j F_j."""
    M = np.asarray(M0, float).copy()
    for j in np.where(selected)[0]:
        M += float(w_pkt[j]) * F[j]
    return M


def acquire(
    F, c, M0, Sigma_pre, Sigmas, B, rho, sigma2, beta, s_rows,
    seller_ids, sources, Xs, ys, qs, X0, y0, lam, kind, kind_arr,
    z_idx, rng, reuse=True, use_quality=True,
    oracle_q=False, random_trial=False, allow_sellers=None,
    ys_clean=None, yids=None,
    quality_mode="qsource", oracle_train=False,
    crossfit=True, seed_only_ref=False, skip_final_screen=False,
    n_trials_override=None, skip_trial_sellers=None, pi_override=None,
    extra_trial_sellers=None, extra_trial_budget=0.0,
    pi_plan=None,
    exhaust_budget=False,
    platform_trial_idx=None,
    trial_trace=None,
):
    """Fixed-β purchase: trial ranking, contamination-weighted design, packet fill.

    π̂_j is the Beta(2,1) posterior mean. Design uses seller π̂_j; training uses
    per-row clean posteriors. n_trials_override replaces floor(βB / trial cost).
    platform_trial_idx are platform-paid trial offers (not charged to B).
    trial_trace: if a list, append trial and budget records.

    reuse=False: trials used only for π̂, not in train / not in M.
    use_quality=False: weights are 1.
    random_trial=True: uniform seller coverage.
    """
    from qsource.quality import (
        QUALITY_ROUNDS,
        estimate_on_mask,
        obs_dirty_mask,
        oracle_clean_weights,
        packet_row_weights_from_gamma,
        seed_sigma_c2,
    )

    seller_ids = np.asarray(seller_ids)
    kind_arr = np.asarray(kind_arr)
    is_trial = kind_arr != KIND_PACKET
    unique = [int(s) for s in np.unique(seller_ids)]
    trial_cost = float(s_rows) / float(PKT_ROWS)
    if n_trials_override is None:
        n_trials = max(1, int(np.floor(beta * B / max(trial_cost, 1e-9))))
    else:
        n_trials = max(0, int(n_trials_override))
    if allow_sellers is not None:
        allow_sellers = {int(x) for x in allow_sellers}
    skip_trial_sellers = {int(x) for x in (skip_trial_sellers or [])}
    ranked = rank_sellers(F, M0, Sigma_pre, sigma2, seller_ids, kind_arr)
    if random_trial:
        sellers = [s for s, _ in ranked]
        rng.shuffle(sellers)
        ranked = [(s, 0.0) for s in sellers]
    if allow_sellers is not None:
        ranked = [(s, g) for s, g in ranked if int(s) in allow_sellers]
    ranked_trial = [(s, g) for s, g in ranked if int(s) not in skip_trial_sellers]

    trial_mask = np.zeros(len(c), bool)
    q_est, q_true_chk, traj = {}, {}, []
    remaining, trial_spend, n_checked = float(B), 0.0, 0
    for s, g in ranked_trial[:n_trials]:
        pj = idx_of(seller_ids, kind_arr, s, KIND_TRIAL)
        if pj < 0 or c[pj] > remaining + 1e-9:
            continue
        trial_mask[pj] = True
        remaining -= c[pj]
        trial_spend += c[pj]
        n_checked += 1
        q_true_chk[s] = float(qs[pj])
        traj.append(dict(
            step=len(traj), phase="purchase", remaining_B=remaining,
            chosen_seller=int(s), chosen_pkt=int(pj), is_trial=True,
            q_est=np.nan, q_true=float(qs[pj]), score=float(g),
        ))
    extra_trial_budget = float(extra_trial_budget or 0.0)
    if extra_trial_budget > 1e-12:
        extra_rem = extra_trial_budget
        already = {int(s) for s in q_true_chk}
        for s in extra_trial_sellers or []:
            s = int(s)
            if extra_rem < trial_cost - 1e-9:
                break
            if s in already:
                continue
            if allow_sellers is not None and s not in allow_sellers:
                continue
            pj = idx_of(seller_ids, kind_arr, s, KIND_TRIAL)
            if pj < 0 or c[pj] > remaining + 1e-9:
                continue
            trial_mask[pj] = True
            remaining -= c[pj]
            extra_rem -= float(c[pj])
            trial_spend += c[pj]
            n_checked += 1
            already.add(s)
            q_true_chk[s] = float(qs[pj])
            traj.append(dict(
                step=len(traj), phase="explore", remaining_B=remaining,
                chosen_seller=int(s), chosen_pkt=int(pj), is_trial=True,
                q_est=np.nan, q_true=float(qs[pj]), score=0.0,
            ))

    dataset = "amazon" if kind == "regression" else "domainnet"
    C = int(np.asarray(y0).shape[1]) if np.asarray(y0).ndim == 2 else None
    seller_w = {s: 1.0 for s in unique}
    row_w = {}
    sigma_c2 = seed_sigma_c2(X0, y0, lam) if dataset == "amazon" else None
    use_invvar = bool(use_quality and quality_mode == "invvar")

    if use_invvar and n_checked >= 1:
        raise NotImplementedError("invvar quality mode is not part of QSource")
    elif use_quality and n_checked >= 1 and not oracle_q:
        pi_trial, _, sigma_c2, _ = estimate_on_mask(
            dataset, seller_ids, sources, trial_mask, Xs, ys, yids, ys_clean,
            X0, y0, lam, sigma_c2=sigma_c2, C=C, n_rounds=1,
            leave_seller_out=crossfit, seed_only_ref=seed_only_ref,
        )
        pi_mean = float(np.mean(list(pi_trial.values()))) if pi_trial else 0.5
        seller_w = {s: float(pi_trial.get(s, pi_mean)) for s in unique}
        for s in q_true_chk:
            q_est[s] = float(seller_w.get(int(s), 1.0))
        for tr in traj:
            s = int(tr["chosen_seller"])
            tr["q_est"] = float(seller_w.get(s, 1.0))
            tr["w"] = float(seller_w.get(s, 1.0))
            tr["pi_hat"] = float(seller_w.get(s, 1.0))
    elif use_quality and n_checked >= 1 and oracle_q and ys_clean is not None:
        for s in q_true_chk:
            pj = idx_of(seller_ids, kind_arr, s, KIND_TRIAL)
            if pj < 0:
                continue
            dirty = obs_dirty_mask(ys[pj], ys_clean[pj])
            seller_w[int(s)] = float(1.0 - dirty.mean())
            q_est[s] = seller_w[int(s)]
        pi_mean = float(np.mean([seller_w[s] for s in q_true_chk])) if q_true_chk else 0.5
        for s in unique:
            if s not in q_true_chk:
                seller_w[s] = pi_mean
        for tr in traj:
            s = int(tr["chosen_seller"])
            tr["q_est"] = float(seller_w.get(s, 1.0))
            tr["w"] = float(seller_w.get(s, 1.0))

    trial_sids = {int(s) for s in q_true_chk}
    if pi_override:
        for s, pval in pi_override.items():
            s = int(s)
            if s in trial_sids:
                continue
            seller_w[s] = float(np.clip(float(pval), 1e-3, 1.0 - 1e-3))
            q_est[s] = seller_w[s]

    w_pkt = np.array([float(seller_w.get(int(s), 1.0)) for s in seller_ids], float)
    if pi_plan:
        for j, s in enumerate(seller_ids):
            s = int(s)
            if s in trial_sids:
                continue
            if s in pi_plan:
                w_pkt[j] = float(np.clip(float(pi_plan[s]), 1e-3, 1.0 - 1e-3))
    if not use_quality:
        w_pkt[:] = 1.0
        seller_w = {s: 1.0 for s in unique}

    Fq = F * w_pkt[:, None, None]
    pk = ~is_trial
    alpha = best_alpha(Fq[pk], c[pk], M0, Sigma_pre, Sigmas, B, rho, sigma2)
    B1 = float(np.floor(alpha * B))

    def _fill(Sigma, budget_cap, selected0):
        selected = np.asarray(selected0, bool).copy()
        while True:
            spent = float(c[selected].sum()) if selected.any() else 0.0
            remaining = float(budget_cap) - spent
            if remaining < 1e-9:
                break
            M = _M_weighted(F, M0, selected, w_pkt)
            try:
                loss = _risk(Sigma, M, sigma2)
            except np.linalg.LinAlgError:
                break
            best_j, best_score = -1, 0.0
            for j in range(len(c)):
                if selected[j] or c[j] > remaining + 1e-9 or is_trial[j]:
                    continue
                s = int(seller_ids[j])
                if allow_sellers is not None and s not in allow_sellers:
                    continue
                wj = float(w_pkt[j])
                try:
                    gain = loss - _risk(Sigma, M + wj * F[j], sigma2)
                except np.linalg.LinAlgError:
                    continue
                score = gain / c[j]
                if score > best_score:
                    best_score, best_j = score, j
            if best_j < 0 or best_score <= 0:
                break
            selected[best_j] = True
        return selected

    n_main = int(n_checked)
    main_sellers = tuple(int(s) for s in q_true_chk)

    if reuse:
        selected = trial_mask.copy()
        spent = float(c[selected].sum())
        s1 = _fill(Sigma_pre, max(B1, spent), selected)
        s2 = _fill(Sigmas[z_idx], B, s1)
    else:
        selected = np.zeros(len(c), bool)
        B1_pkt = max(B1 - trial_spend, 0.0)
        B_pkt = max(B - trial_spend, 0.0)
        s1 = _fill(Sigma_pre, B1_pkt, selected)
        s2 = _fill(Sigmas[z_idx], B_pkt, s1)

    pkt_mask = s2 & ~is_trial
    rem_after_fill = float(B) - float(c[s2].sum())
    packet_spend_fill = float(c[pkt_mask].sum()) if pkt_mask.any() else 0.0
    extra_sellers = []

    if exhaust_budget:
        # Leftover is < 1 when n_trials * trial_cost is not integer in B.
        # Spend it on unused trials so instance totals hit B per buyer.
        while True:
            remaining = float(B) - float(c[s2].sum())
            if remaining < trial_cost - 1e-9:
                break
            added = False
            for s, g in ranked_trial:
                pj = idx_of(seller_ids, kind_arr, s, KIND_TRIAL)
                if pj < 0 or s2[pj] or c[pj] > remaining + 1e-9:
                    continue
                if allow_sellers is not None and int(s) not in allow_sellers:
                    continue
                if trial_trace is not None:
                    trial_trace.append(dict(
                        kind="exhaust",
                        z_idx=int(z_idx),
                        extra_i=int(len(extra_sellers) + 1),
                        seller=int(s),
                        rem_before=float(remaining),
                        trial_cost=float(trial_cost),
                    ))
                extra_sellers.append(int(s))
                s2[pj] = True
                trial_mask[pj] = True
                trial_spend += float(c[pj])
                n_checked += 1
                q_true_chk[int(s)] = float(qs[pj])
                added = True
                break
            if not added:
                break

    if trial_trace is not None:
        trial_trace.append(dict(
            kind="demand_summary",
            z_idx=int(z_idx),
            n_main=int(n_main),
            n_extra=int(len(extra_sellers)),
            n_checked=int(n_checked),
            trial_spend=float(trial_spend),
            packet_spend=float(packet_spend_fill),
            rem_after_fill=float(rem_after_fill),
            rem_after_exhaust=float(B) - float(c[s2].sum()),
            B=float(B),
            main_sellers=main_sellers,
            extra_sellers=tuple(extra_sellers),
        ))

    if platform_trial_idx:
        for pj in platform_trial_idx:
            pj = int(pj)
            if pj < 0 or pj >= len(c):
                continue
            s2[pj] = True
            trial_mask[pj] = True
            s = int(seller_ids[pj])
            if s not in q_true_chk:
                n_checked += 1
                q_true_chk[s] = float(qs[pj])

    if use_quality and (oracle_q or oracle_train) and ys_clean is not None:
        row_w = oracle_clean_weights(s2, ys, ys_clean)
    elif use_invvar and n_checked >= 1:
        row_w = {
            int(j): float(seller_w.get(int(seller_ids[j]), 1.0))
            for j in np.where(s2)[0]
        }
    elif use_quality and not skip_final_screen and (
        n_checked >= 1 or (pi_override and bool(np.any(s2)))
    ):
        n_final = 1 if seed_only_ref else QUALITY_ROUNDS
        pi_final, gmap, _, _ = estimate_on_mask(
            dataset, seller_ids, sources, s2, Xs, ys, yids, ys_clean,
            X0, y0, lam, sigma_c2=sigma_c2, C=C, n_rounds=n_final,
            leave_seller_out=crossfit, seed_only_ref=seed_only_ref,
        )
        row_w = packet_row_weights_from_gamma(seller_ids, s2, Xs, gmap, pi_final)
    else:
        row_w = {j: 1.0 for j in np.where(s2)[0]}

    return s2, alpha, traj, q_est, q_true_chk, n_checked, trial_spend, trial_mask, seller_w, row_w
