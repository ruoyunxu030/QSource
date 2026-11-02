"""Run one market instance: buyers, optional injection, shared store, records."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from qsource.attacks import (
    injection_value,
    insert_after,
    n_accounts,
    pick_target,
    published_mean_and_sigmahat,
    seller_pi_map,
    write_injected,
)
from qsource.buyer import (
    buyer_pi_hats,
    draw_buyer_task,
    run_one_buyer,
    visible_market,
)
from qsource.const import (
    B,
    BACKED_SOURCES,
    BUYER_DESIGNS,
    BUYER_SALT,
    CAL_PRIOR,
    CLIP_BUYER,
    DELTA,
    DS_ID,
    ETA_DEFAULT,
    ETA_ONE,
    FLOOR_DESIGNS,
    FLOOR_ROWS,
    FLOOR_SALT,
    K,
    KIND_BUYER,
    KIND_ROW,
    L,
    LOCK,
    MEDIOCRE,
    N_BUYERS,
    N_SEED,
    PHASE_TEST,
    PKT_ROWS,
    POLICY_SEED_ID,
    RAISE_WEAK,
    RHO,
    ROW_DESIGNS,
    SEED_BASE,
    SKIP_THRESHOLD,
    SRC_BACKED,
    SRC_PLATFORM,
    SRC_TRIAL,
    W_MAX,
)
from qsource.floor import draw_platform_seed, platform_rng, score_floor_offers
from qsource.market import append_floor_trials, build_market, n_extra_trials
from qsource.store import (
    ArrivalSkipStore,
    BuyerEvidenceStore,
    EvidenceStore,
    PurchaseWeightStore,
    QSourceStore,
    Record,
    RobustBuyerStore,
    classify_source,
    floor_oid_set,
)


def cfg_of(dataset: str, inst: int) -> dict:
    lock = LOCK[dataset]
    return dict(
        dataset=dataset, beta=lock["beta"], s_rows=lock["s_rows"],
        B=B, K=K, rho=RHO, n_seed=N_SEED, inst=int(inst),
        sampling_rule=lock["sampling_rule"],
    )


def market_rng(dataset, s_rows, B, inst):
    return np.random.default_rng(
        [SEED_BASE, DS_ID[dataset], int(s_rows), int(B), int(inst), PHASE_TEST]
    )


def floor_rng(dataset, s_rows, B, inst):
    return np.random.default_rng(
        [SEED_BASE, DS_ID[dataset], int(s_rows), int(B), int(inst),
         PHASE_TEST, FLOOR_SALT]
    )


def buyer_rng(dataset, s_rows, B, inst, buyer_t):
    return np.random.default_rng(
        [SEED_BASE, DS_ID[dataset], int(s_rows), int(B), int(inst),
         PHASE_TEST, int(buyer_t), BUYER_SALT]
    )


def attack_spec(config: str, dataset: str, timing: str = "early") -> dict:
    if config == "none":
        return dict(kind="none", variant="none")
    if config == "raise_weak":
        return dict(
            kind="injection", eta_b=ETA_DEFAULT, goal="raise",
            value_mode="raise_weak", timing=str(timing), variant="raise_weak",
        )
    eta = ETA_ONE if config == "extreme_one_account" else ETA_DEFAULT
    return dict(
        kind="injection", eta_b=eta, goal="lower",
        value_mode="extreme" if config in ("extreme", "extreme_one_account") else config,
        timing=str(timing), variant=config,
    )


def method_flags(method: str) -> dict:
    if method == "qsource":
        return dict(apply_qsource=True, skip_in_risk=True, unw_when_no_backed=True,
                    value_filter=False, robust=None)
    if method in ("forced_trials", "forced_repeat_trials"):
        return dict(apply_qsource=True, skip_in_risk=False, unw_when_no_backed=True,
                    value_filter=False, robust=None)
    if method == "value_filter":
        return dict(apply_qsource=False, skip_in_risk=False, unw_when_no_backed=False,
                    value_filter=True, robust=None)
    if method == "median":
        return dict(apply_qsource=False, skip_in_risk=False, unw_when_no_backed=False,
                    value_filter=False, robust="median")
    if method == "trimmed_mean":
        return dict(apply_qsource=False, skip_in_risk=False, unw_when_no_backed=False,
                    value_filter=False, robust="trim")
    if method == "purchase_weight":
        return dict(apply_qsource=False, skip_in_risk=False, unw_when_no_backed=False,
                    value_filter=False, robust=None, purchase_weight=True)
    return dict(apply_qsource=False, skip_in_risk=False, unw_when_no_backed=False,
                value_filter=False, robust=None)


def _rids_json(rids) -> str:
    return json.dumps([[int(a), int(b)] for a, b in (rids or ())])


def _floor_idx(vis, market, design, ej, used_oids, s_rows):
    if design not in FLOOR_DESIGNS:
        return None
    F = FLOOR_ROWS[design]
    loc = {int(o): i for i, o in enumerate(np.asarray(vis["offer_id"], int))}
    by_s = market.get("floor_offer_ids") or {}
    idx = []
    vis_sids = sorted({int(s) for s in vis["seller_ids"]})
    for s in vis_sids:
        have = len(ej.get(int(s), set()))
        need = int(F) - have
        if need <= 0:
            continue
        n = int(math.ceil(need / float(s_rows)))
        oids = [int(o) for o in by_s.get(int(s), []) if int(o) not in used_oids]
        take = oids[:n]
        for o in take:
            used_oids.add(int(o))
            if int(o) in loc:
                idx.append(loc[int(o)])
    return idx or None


def _strip_floor(rids, gs, floor_oids: set[int]):
    if not floor_oids:
        return tuple(rids or ()), tuple(gs or ())
    keep_r, keep_g = [], []
    for rid, g in zip(rids or (), gs or ()):
        if int(rid[0]) not in floor_oids:
            keep_r.append(rid)
            keep_g.append(g)
    return tuple(keep_r), tuple(keep_g)


def _rids_of_seller(gamma_by_row, vis, sid: int):
    offer_id = np.asarray(vis["offer_id"], int)
    seller_ids = np.asarray(vis["seller_ids"], int)
    loc = {int(oid): i for i, oid in enumerate(offer_id)}
    out, gs = [], []
    for rid, g in gamma_by_row.items():
        oid = int(rid[0])
        j = loc.get(oid)
        if j is None:
            continue
        if int(seller_ids[j]) != int(sid):
            continue
        out.append((oid, int(rid[1])))
        gs.append(float(g))
    return tuple(out), tuple(gs)


def run_instance(
    bundle,
    dataset: str,
    design: str,
    method: str,
    config: str = "none",
    timing: str = "early",
    inst: int = 230,
    n_buyers: int = N_BUYERS,
    q01: float | None = None,
    w_max: float = W_MAX,
    delta: float = DELTA,
    skip_threshold: float = SKIP_THRESHOLD,
    target_goal: str | None = None,
):
    flags = method_flags(method)
    spec = attack_spec(config, dataset, timing)
    variant = str(spec.get("variant") or "none")
    kind = str(spec.get("kind") or "none")
    goal = str(target_goal or spec.get("goal") or "lower")
    buy_packet = bool(spec.get("buy_packet"))
    eta_b = float(spec.get("eta_b") or 0.0)
    cfg = cfg_of(dataset, inst)
    cfg.update(lam=bundle["lam"], sigma2=bundle["sigma2"], d=bundle["d"])
    t0 = time.time()
    groups = bundle["groups"]
    n_extra = n_extra_trials(cfg["s_rows"])
    market = build_market(
        market_rng(dataset, cfg["s_rows"], cfg["B"], inst),
        groups, dataset, cfg["s_rows"], n_sellers=8,
    )
    market = append_floor_trials(
        market, floor_rng(dataset, cfg["s_rows"], cfg["B"], inst),
        groups, dataset, cfg["s_rows"], n_extra,
    )
    plat_seed = None
    if design in FLOOR_DESIGNS:
        rng_plat = platform_rng(dataset, cfg["s_rows"], cfg["B"], inst)
        plat_seed = draw_platform_seed(rng_plat, groups, market, cfg)
    target = pick_target(market, goal)
    n_atk = n_accounts(eta_b, n_buyers) if kind == "injection" else 0
    inserts = insert_after(eta_b, spec.get("timing") or "early", n_buyers) if kind == "injection" else {}
    seller_of = {int(oid): int(s) for oid, s in zip(market["offer_id"], market["seller_ids"])}
    src_of = {int(s): str(g) for s, g in zip(market["seller_ids"], market["sources"])}
    pis = seller_pi_map(market)
    floor_oids = floor_oid_set(market)
    apply_qs = bool(flags["apply_qsource"])
    robust = flags.get("robust")
    row_store = EvidenceStore(seller_of) if design in ROW_DESIGNS else None
    if design in BUYER_DESIGNS:
        prior = CAL_PRIOR.get(design, "eb")
        if flags.get("purchase_weight"):
            buyer_store = PurchaseWeightStore(prior=prior, pkt_rows=int(PKT_ROWS))
        elif robust:
            buyer_store = RobustBuyerStore(prior=prior, agg=str(robust))
        elif abs(float(skip_threshold) - 1.0) > 1e-12:
            buyer_store = ArrivalSkipStore(prior=prior)
        else:
            buyer_store = BuyerEvidenceStore(prior=prior)
    else:
        buyer_store = None
    sidecar = BuyerEvidenceStore(prior="eb")
    qs_kind = KIND_ROW if design in ROW_DESIGNS else KIND_BUYER
    qs_store = QSourceStore(
        kind=qs_kind, w_max=float(w_max), delta=float(delta),
        pkt_rows=PKT_ROWS, prior=CAL_PRIOR.get(design, "eb"),
        unw_when_no_backed=bool(flags["unw_when_no_backed"]),
        skip_in_risk=bool(flags["skip_in_risk"]),
        skip_threshold=float(skip_threshold),
    ) if apply_qs else None
    ej = {}
    floor_used = set()
    buyer_rows, pay_rows, post_rows, traj_rows = [], [], [], []
    rec_rows, plat_rows = [], []
    all_sids = sorted({int(x) for x in market["seller_ids"]})
    atk_seq = 0
    true_pi_tgt = float(pis[int(target)]) if target is not None else None
    trial_cost = float(cfg["s_rows"]) / float(PKT_ROWS)
    L_ds = float(L[dataset])
    mediocre_value = float(MEDIOCRE[dataset])
    filt = bool(flags["value_filter"])
    if filt and (q01 is None or not np.isfinite(float(q01))):
        raise SystemExit(
            f"value_filter requires a threshold for dataset={dataset} "
            f"design={design} instance={inst}; missing threshold, not disabling the filter"
        )
    cell = dict(
        dataset=dataset, design=design, method=method,
        config=str(config),
        timing=str(timing), instance=int(inst),
        skip_threshold=float(skip_threshold),
        target_goal=str(goal),
    )

    def _pub_now():
        if apply_qs:
            return {int(s): qs_store.published_clip(int(s)) for s in all_sids}
        if buyer_store is not None:
            out = {}
            for s in all_sids:
                m, _v, n, *_ = buyer_store.publish(int(s))
                out[int(s)] = float(np.clip(m, CLIP_BUYER[0], CLIP_BUYER[1]))
            return out
        if row_store is not None:
            out = {}
            for s in all_sids:
                _a, _b, mean, n = row_store.seller_posterior(int(s))
                out[int(s)] = float(mean)
            return out
        return {int(s): float("nan") for s in all_sids}

    def _risk_now(sid):
        if apply_qs:
            return int(qs_store.in_risk(int(sid)))
        return 0

    def _do_insert(after_t: int, vis_now):
        nonlocal atk_seq
        n_ins = int(inserts.get(int(after_t), 0))
        if kind != "injection" or target is None or n_ins <= 0:
            return
        for _k in range(n_ins):
            atk_seq += 1
            if apply_qs:
                m_j = qs_store.published(int(target)) if (qs_store.n_backed(int(target)) + qs_store.n_trial(int(target))) else (2.0 / 3.0)
                sig = qs_store.sigmahat()
            else:
                m_j, sig = published_mean_and_sigmahat(buyer_store, row_store, sidecar, int(target))
            if config == "raise_weak":
                val = float(RAISE_WEAK[dataset])
            else:
                val = injection_value(variant, m_j, sig, L_ds, mediocre_value)
            wr = write_injected(vis_now, int(target), val, buy_packet=buy_packet)
            paid = float(wr.get("paid") or 0.0)
            bought = int(int(wr.get("packets_bought") or 0) >= 1)
            src = SRC_BACKED if bought else SRC_TRIAL
            drop = False
            if apply_qs:
                rec = Record(
                    seller=int(target), pi_hat=float(val),
                    rids=tuple(wr.get("rids") or ()),
                    gammas=tuple(wr.get("gammas") or ()),
                    source=src, t=int(after_t), buyer_id=-int(atk_seq),
                    n_rows=int(wr.get("n_rows") or 0),
                    packets_bought=int(wr.get("packets_bought") or 0),
                    paid=paid,
                )
                if wr.get("ok"):
                    qs_store.add(rec)
            else:
                drop = bool(filt and np.isfinite(q01) and float(val) < float(q01))
                if wr.get("ok") and not drop:
                    if buyer_store is not None:
                        buyer_store.update_buyer(int(target), float(val), int(wr.get("n_rows") or 0), after_t)
                    if row_store is not None:
                        rids = wr.get("rids") or ()
                        row_store.update({rid: float(val) for rid in rids})
                    # sidecar is honest-only σ̂. Used for injection σ when there is
                    # no buyer store (row-level). Attacker writes never go here,
                    # early or late, on any non-QSource path. Buyer-level injection
                    # reads σ from the buyer store, which does include attacks.
                    # Floor counts only backed rows; injected trial-only rows
                    # do not fill E_backed.
                    if src in BACKED_SOURCES:
                        for rid in wr.get("rids") or ():
                            ej.setdefault(int(target), set()).add(rid)
            rec_rows.append(dict(
                **cell, buyer=int(after_t), seller=int(target),
                pi_hat=float(val), n_rows=int(wr.get("n_rows") or 0),
                source=src,
                is_attacker=1, packets=float(wr.get("packets_bought") or 0),
                bought_packet=bought, rids_json=_rids_json(wr.get("rids") or ()),
                in_risk=_risk_now(int(target)),
                published=qs_store.published(int(target)) if apply_qs else _pub_now().get(int(target), np.nan),
            ))
            if wr.get("ok"):
                pay_rows.append(dict(
                    **cell, buyer=int(after_t), seller=int(target), is_attacker=1,
                    true_pi=true_pi_tgt, paid=paid,
                    packets_bought=int(wr.get("packets_bought") or 0),
                    bought_packet=bought, n_trials=0.0, excess=np.nan,
                    packet_paid=0.0, platform_paid=0.0,
                ))
            buyer_rows.append(dict(
                **cell, buyer=int(after_t), is_attacker=1,
                excess=np.nan, n_trials=0.0,
                packets=float(wr.get("packets_bought") or 0),
                platform_paid=0.0,
            ))
            pubs = _pub_now()
            for s in all_sids:
                post_rows.append(dict(
                    **cell, buyer=int(after_t), seller=int(s), is_attacker=1,
                    published=pubs[int(s)], in_risk=_risk_now(s),
                    true_pi=float(pis.get(int(s), np.nan)),
                ))

    def _keep_hat(pi_hat) -> bool:
        return not (filt and q01 is not None and np.isfinite(float(q01)) and float(pi_hat) < float(q01))

    def _do_floor(t, vis):
        ej_now = qs_store.backed_row_ids() if apply_qs else ej
        plat_idx = _floor_idx(vis, market, design, ej_now, floor_used, cfg["s_rows"])
        skip_s = {int(vis["seller_ids"][j]) for j in plat_idx} if plat_idx else None
        if plat_idx and plat_seed is not None:
            scored = score_floor_offers(
                vis, plat_idx, plat_seed["X0"], plat_seed["y0"], dataset, cfg["lam"],
            )
            for s, sc in scored.items():
                rec = Record(
                    seller=int(s), pi_hat=float(sc["pi_hat"]),
                    rids=tuple(sc["rids"]), gammas=tuple(sc["gammas"]),
                    source=SRC_PLATFORM, t=int(t), buyer_id=0,
                    n_rows=int(sc["n_rows"]), packets_bought=0,
                    paid=float(sc["paid"]),
                )
                keep = _keep_hat(rec.pi_hat)
                if keep:
                    if apply_qs:
                        qs_store.add(rec)
                    else:
                        if buyer_store is not None:
                            if isinstance(buyer_store, PurchaseWeightStore):
                                buyer_store.update_buyer(
                                    int(s), float(rec.pi_hat), int(rec.n_rows), int(t),
                                    weight=1.0,
                                )
                            else:
                                buyer_store.update_buyer(
                                    int(s), float(rec.pi_hat), int(rec.n_rows), int(t),
                                )
                        ej.setdefault(int(s), set()).update(rec.rids or ())
                rec_rows.append(dict(
                    **cell, buyer=int(t), seller=int(s),
                    pi_hat=float(rec.pi_hat), n_rows=int(rec.n_rows),
                    source=SRC_PLATFORM,
                    is_attacker=0, packets=0.0, bought_packet=0,
                    rids_json=_rids_json(rec.rids),
                    in_risk=_risk_now(int(s)),
                    published=(
                        qs_store.published(int(s)) if apply_qs
                        else _pub_now().get(int(s), np.nan)
                    ),
                ))
                plat_rows.append(dict(
                    **cell, buyer=int(t), seller=int(s),
                    platform_paid=float(sc["paid"]),
                    platform_rows=int(sc["n_rows"]),
                ))
        return plat_idx, skip_s

    vis0 = visible_market(market, 1)
    _do_insert(0, vis0)
    for t in range(1, n_buyers + 1):
        vis = visible_market(market, t)
        vis_sids = sorted({int(s) for s in vis["seller_ids"]})
        b_rng = buyer_rng(dataset, cfg["s_rows"], cfg["B"], inst, t)
        demand, Sigmas, Sigma_pre, M0, X0, y0, X_mix, y_mix, yid_mix = draw_buyer_task(
            b_rng, groups, market, cfg["n_seed"], cfg["d"], cfg["lam"], cfg["K"],
        )
        pol_rng = np.random.default_rng(
            [SEED_BASE, DS_ID[dataset], int(cfg["s_rows"]), int(cfg["B"]),
             int(inst), PHASE_TEST, int(t), BUYER_SALT, POLICY_SEED_ID.get(design, 9)]
        )
        plat_idx, skip_s = _do_floor(t, vis)
        floor_oid_now = (
            {int(vis["offer_id"][j]) for j in plat_idx} if plat_idx else set()
        )
        qs_skip = qs_n_trials = qs_pi = None
        store = row_store
        bstore = buyer_store
        pubs_before = _pub_now()
        risk_before = {int(s): _risk_now(s) for s in vis_sids}
        if apply_qs:
            qs_n_trials, skip = qs_store.skip_and_ntrials(vis_sids, cfg["B"], cfg["beta"], trial_cost)
            qs_skip = skip
            qs_pi = qs_store.pi_map(vis_sids)
            store = None
            bstore = None
        summary, payments, gamma_by_row, _pkt = run_one_buyer(
            design, vis, groups, demand, Sigmas, Sigma_pre,
            M0, X0, y0, X_mix, y_mix, yid_mix, cfg, pol_rng, store,
            buyer_store=bstore,
            exhaust_budget=design != "known_quality",
            platform_trial_idx=plat_idx, extra_skip_sellers=skip_s,
            qs_skip=qs_skip, qs_n_trials=qs_n_trials, qs_pi=qs_pi,
            skip_threshold=float(skip_threshold),
        )
        if gamma_by_row and floor_oid_now:
            gamma_by_row = {
                rid: g for rid, g in gamma_by_row.items()
                if int(rid[0]) not in floor_oid_now
            }
        hats = buyer_pi_hats(gamma_by_row, vis) if gamma_by_row else {}
        pay_by_s = {int(r["seller"]): r for r in payments}
        bought_any = int(any(int(r.get("bought_packet") or 0) >= 1 for r in payments))
        tgt_packets = 0.0
        if target is not None and int(target) in pay_by_s:
            tgt_packets = float(pay_by_s[int(target)].get("packets_bought") or 0.0)
        keep_hats = {}
        for s, (ph, nr) in hats.items():
            if not _keep_hat(ph):
                continue
            keep_hats[int(s)] = (ph, nr)
        if apply_qs and gamma_by_row:
            for s, (ph, nr) in hats.items():
                rids, gs = _rids_of_seller(gamma_by_row, vis, s)
                rids, gs = _strip_floor(rids, gs, floor_oid_now)
                if not rids:
                    continue
                pay = pay_by_s.get(int(s), {})
                pkt_avg = float(pay.get("packets_bought") or 0.0)
                bought = int(pay.get("bought_packet") or 0)
                src = classify_source(
                    bought, rids, floor_oids.get(int(s), set()),
                    bought_any_packet=bought_any,
                )
                rec = Record(
                    seller=int(s), pi_hat=float(ph), rids=rids, gammas=gs,
                    source=src, t=int(t), buyer_id=int(t), n_rows=int(len(gs)),
                    packets_bought=pkt_avg,
                    paid=float(pay.get("paid") or 0.0),
                )
                qs_store.add(rec)
                rec_rows.append(dict(
                    **cell, buyer=int(t), seller=int(s),
                    pi_hat=float(ph), n_rows=int(len(gs)),
                    source=src,
                    is_attacker=0, packets=pkt_avg, bought_packet=bought,
                    rids_json=_rids_json(rids),
                    in_risk=int(qs_store.in_risk(int(s))),
                    published=qs_store.published(int(s)),
                ))
        elif not apply_qs:
            if store is not None and gamma_by_row:
                if filt:
                    offer_id = np.asarray(vis["offer_id"], int)
                    seller_ids = np.asarray(vis["seller_ids"], int)
                    loc = {int(oid): i for i, oid in enumerate(offer_id)}
                    kept = {}
                    for rid, g in gamma_by_row.items():
                        j = loc.get(int(rid[0]))
                        if j is None:
                            continue
                        s = int(seller_ids[j])
                        if s in keep_hats:
                            kept[rid] = g
                    if kept:
                        store.update(kept)
                elif gamma_by_row:
                    store.update(gamma_by_row)
            if gamma_by_row:
                for s, (ph, nr) in keep_hats.items():
                    sidecar.update_buyer(s, ph, nr, t)
                    rids, gs = _rids_of_seller(gamma_by_row, vis, s)
                    rids, gs = _strip_floor(rids, gs, floor_oid_now)
                    if not rids:
                        continue
                    pay = pay_by_s.get(int(s), {})
                    bought = int(pay.get("bought_packet") or 0)
                    src = classify_source(
                        bought, rids, floor_oids.get(int(s), set()),
                        bought_any_packet=bought_any,
                    )
                    if src in BACKED_SOURCES:
                        for rid in rids:
                            ej.setdefault(int(s), set()).add(rid)
                    if bstore is not None:
                        if isinstance(bstore, PurchaseWeightStore):
                            w = 1.0 if src in BACKED_SOURCES else min(
                                1.0, float(len(gs)) / float(PKT_ROWS),
                            )
                            bstore.update_buyer(s, ph, len(gs), t, weight=w)
                        else:
                            bstore.update_buyer(s, ph, len(gs), t)
                    rec_rows.append(dict(
                        **cell, buyer=int(t), seller=int(s),
                        pi_hat=float(ph), n_rows=int(len(gs)),
                        source=src,
                        is_attacker=0, packets=float(pay.get("packets_bought") or 0.0),
                        bought_packet=bought, rids_json=_rids_json(rids),
                        in_risk=_risk_now(int(s)),
                        published=_pub_now().get(int(s), np.nan),
                    ))
        buyer_rows.append(dict(
            **cell, buyer=int(t), is_attacker=0,
            excess=summary["excess"], n_trials=summary["n_checked"],
            trial_spend=float(summary.get("trial_spend") or 0.0),
            packets=tgt_packets,
            platform_paid=float(summary.get("platform_paid") or 0.0),
        ))
        for rec in payments:
            s = int(rec["seller"])
            pay_rows.append(dict(
                **cell, buyer=int(t), seller=s, is_attacker=0,
                true_pi=float(rec["true_pi"]), paid=float(rec["paid"]),
                packets_bought=float(rec["packets_bought"]),
                bought_packet=int(rec.get("bought_packet") or 0),
                n_trials=summary["n_checked"], excess=summary["excess"],
                packet_paid=float(rec.get("packet_paid") or 0.0),
                platform_paid=float(rec.get("platform_paid") or 0.0),
            ))
        pubs = _pub_now()
        for s in all_sids:
            post_rows.append(dict(
                **cell, buyer=int(t), seller=int(s), is_attacker=0,
                published=pubs[int(s)], in_risk=_risk_now(s),
                true_pi=float(pis.get(int(s), np.nan)),
            ))
        if target is not None:
            traj_rows.append(dict(
                **cell, buyer=int(t),
                packets=tgt_packets,
                published=pubs_before.get(int(target), np.nan),
                in_risk=risk_before.get(int(target), 0),
                n_trials=summary["n_checked"],
                excess=summary["excess"],
            ))
        print(
            f"  {dataset} inst{inst} {design}/{method}/{variant} buyer {t}/{n_buyers}",
            flush=True,
        )
        _do_insert(t, vis)

    meta = dict(
        **cell, n_buyers=n_buyers, seconds=time.time() - t0,
        target_seller=target, n_injected=int(n_atk),
        w_max=float(w_max), delta=float(delta),
        L=L_ds, mediocre_value=mediocre_value,
        q01=None if q01 is None else float(q01),
        variant=variant,
    )
    return dict(
        buyers=pd.DataFrame(buyer_rows),
        pays=pd.DataFrame(pay_rows),
        posts=pd.DataFrame(post_rows),
        traj=pd.DataFrame(traj_rows),
        records=pd.DataFrame(rec_rows) if rec_rows else pd.DataFrame(),
        plat=pd.DataFrame(plat_rows) if plat_rows else pd.DataFrame(),
        meta=meta,
        market=market,
    )
