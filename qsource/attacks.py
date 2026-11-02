"""Injected trial-only records (early / late)."""

from __future__ import annotations

import numpy as np

from qsource.const import EXTREME_K, KIND_PACKET, KIND_TRIAL, N_ATK, N_BUYERS


def n_accounts(eta_b: float, n_honest: int = N_BUYERS) -> int:
    key = float(eta_b)
    if int(n_honest) == 30 and key in N_ATK:
        return int(N_ATK[key])
    if key >= 1.0:
        return int(n_honest)
    return int(round(key * int(n_honest) / max(1.0 - key, 1e-9)))


def insert_after(eta_b: float, timing: str, n_honest: int = N_BUYERS) -> dict[int, int]:
    """How many injected accounts to insert after honest buyer t (t=0 = before buyer 1).

    before_first: all accounts write before buyer 1 (Fig. 1 Mediocre timing).
    """
    n_atk = n_accounts(eta_b, n_honest)
    if n_atk <= 0:
        return {}
    if str(timing) == "before_first":
        return {0: int(n_atk)}
    window = list(range(0, 10)) if str(timing) != "late" else list(range(20, 30))
    counts: dict[int, int] = {}
    for i in range(n_atk):
        t = int(window[i % len(window)])
        counts[t] = counts.get(t, 0) + 1
    return counts


def seller_pi_map(market: dict) -> dict[int, float]:
    sids = np.asarray(market["seller_ids"], int)
    qs = np.asarray(market["qs"], float)
    out = {}
    for s in sorted(set(int(x) for x in sids.tolist())):
        out[int(s)] = float(1.0 - qs[sids == s][0])
    return out


def pick_target(market: dict, goal: str = "lower") -> int | None:
    pi = 0.5 if str(goal) == "raise" else 1.0
    sids = np.asarray(market["seller_ids"], int)
    qs = np.asarray(market["qs"], float)
    cands = [
        s for s in sorted(set(int(x) for x in sids.tolist()))
        if abs(float(1.0 - qs[sids == s][0]) - pi) < 1e-9
    ]
    return int(cands[0]) if cands else None


def extreme_value(m_j: float, sigmahat: float, goal: str = "lower") -> float:
    delta = float(EXTREME_K) * float(sigmahat)
    if str(goal) == "raise":
        return float(np.clip(float(m_j) + delta, 0.0, 1.0))
    return float(np.clip(float(m_j) - delta, 0.0, 1.0))


def injection_value(config: str, m_j: float, sigmahat: float, L: float,
                    mediocre_value: float) -> float:
    """Injected trial value. Low Honest is max(L, published − 2σ)."""
    extreme = extreme_value(m_j, sigmahat, "lower")
    if config == "low_honest":
        return float(max(extreme, float(L)))
    if config == "fixed":
        return 0.5
    if config == "mediocre":
        return float(mediocre_value)
    return float(extreme)


def find_offer(vis: dict, seller: int, kind: str) -> int | None:
    """Same rule as the paper runs: among matching offers, the smallest offer_id."""
    sids = np.asarray(vis["seller_ids"], int)
    kinds = np.asarray(vis["kind"])
    oids = np.asarray(vis["offer_id"], int)
    idx = [
        int(j) for j in range(len(kinds))
        if kinds[j] == kind and int(sids[j]) == int(seller)
    ]
    if not idx:
        return None
    idx.sort(key=lambda j: int(oids[j]))
    return int(idx[0])


def write_injected(vis, seller: int, value: float, buy_packet: bool) -> dict:
    seller = int(seller)
    val = float(value)
    kind = KIND_PACKET if buy_packet else KIND_TRIAL
    j = find_offer(vis, seller, kind)
    if j is None:
        return dict(ok=False, paid=0.0, n_rows=0, offer_id=-1, value=val,
                    rids=(), gammas=(), packets_bought=0)
    paid = float(vis["c"][j])
    oid = int(vis["offer_id"][j])
    n = int(len(vis["Xs"][j]))
    rids = tuple((oid, i) for i in range(n))
    gammas = tuple([val] * n)
    return dict(
        ok=True, paid=float(paid), n_rows=int(n), offer_id=int(oid),
        value=val, rids=rids, gammas=gammas,
        packets_bought=int(buy_packet),
    )


def published_mean_and_sigmahat(bstore, row_store, sidecar, sid: int) -> tuple[float, float]:
    if bstore is not None:
        m, _v, _n, _mu0, _tau, sig2 = bstore.publish(int(sid))
        return float(m), float(np.sqrt(max(float(sig2), 1e-8)))
    m = 2.0 / 3.0
    if row_store is not None:
        _a, _b, m, _n = row_store.seller_posterior(int(sid))
    sig2 = 1.0 / 18.0
    if sidecar is not None:
        _m, _v, _n, _mu0, _tau, sig2 = sidecar.publish(int(sid))
    return float(m), float(np.sqrt(max(float(sig2), 1e-8)))
