"""Platform check: one clean seed, one platform-checked record per inspected seller.

The platform draws a seed of the same size as a buyer seed from the leftover
clean-row pool (disjoint from seller packets). It scores unused floor offers
until each visible seller has F=40 backed rows. Injected trial-only rows do
not count toward that floor.
"""
from __future__ import annotations

import numpy as np

from qsource.const import (
    DS_ID,
    PHASE_TEST,
    PLATFORM_SALT,
    SEED_BASE,
)
from qsource.buyer import draw_buyer_task
from qsource.quality import (
    QUALITY_ROUNDS,
    estimate_on_mask,
    packet_row_weights_from_gamma,
    seed_sigma_c2,
)


def platform_rng(dataset, s_rows, B, inst):
    return np.random.default_rng(
        [SEED_BASE, DS_ID[dataset], int(s_rows), int(B), int(inst),
         PHASE_TEST, PLATFORM_SALT]
    )


def draw_platform_seed(rng, groups, market, cfg) -> dict:
    """One buyer-shaped seed from the leftover pool via draw_buyer_task."""
    n_seed = int(cfg["n_seed"])
    demand, _sig, _pre, M0, X0, y0, _xm, _ym, _yid = draw_buyer_task(
        rng, groups, market, n_seed, cfg["d"], cfg["lam"], cfg["K"],
    )
    return dict(
        demand=list(demand), M0=M0, X0=X0, y0=y0,
        n_seed=int(len(X0)),
    )


def score_floor_offers(vis, plat_idx, X0, y0, dataset, lam):
    """Score floor offers with the same contamination screen as a buyer trial.

    One seed (X0, y0). One mask over all floor offers this round.
    leave_seller_out, QUALITY_ROUNDS, seed_only_ref=False.
    """
    plat_idx = [int(j) for j in plat_idx or [] if int(j) >= 0]
    if not plat_idx:
        return {}
    seller_ids = np.asarray(vis["seller_ids"], int)
    sources = np.asarray(vis["sources"])
    Xs, ys, yids = vis["Xs"], vis["ys"], vis["yids"]
    ys_clean = vis.get("ys_clean")
    offer_id = np.asarray(vis["offer_id"], int)
    c = np.asarray(vis["c"], float)
    mask = np.zeros(len(seller_ids), bool)
    mask[plat_idx] = True
    C = int(np.asarray(y0).shape[1]) if np.asarray(y0).ndim == 2 else None
    sigma_c2 = seed_sigma_c2(X0, y0, lam) if dataset == "amazon" else None
    pi, gmap, _, _ = estimate_on_mask(
        dataset, seller_ids, sources, mask, Xs, ys, yids, ys_clean,
        X0, y0, lam, sigma_c2=sigma_c2, C=C, n_rounds=QUALITY_ROUNDS,
        leave_seller_out=True, seed_only_ref=False,
    )
    row_w = packet_row_weights_from_gamma(seller_ids, mask, Xs, gmap, pi)
    by_s: dict[int, list[int]] = {}
    for j in plat_idx:
        by_s.setdefault(int(seller_ids[j]), []).append(int(j))
    out = {}
    for s, js in by_s.items():
        rids, gs = [], []
        paid = 0.0
        n_rows = 0
        for j in js:
            oid = int(offer_id[j])
            w = np.asarray(row_w.get(int(j), np.ones(len(Xs[j]))), float)
            paid += float(c[j])
            n_rows += int(len(Xs[j]))
            for i, gi in enumerate(w):
                rids.append((oid, int(i)))
                gs.append(float(gi))
        ph = float(np.mean(gs)) if gs else float(pi.get(int(s), np.nan))
        out[int(s)] = dict(
            pi_hat=ph, n_rows=int(n_rows), rids=tuple(rids), gammas=tuple(gs),
            paid=float(paid), n_offers=int(len(js)), n_seed=int(len(X0)),
        )
    return out
