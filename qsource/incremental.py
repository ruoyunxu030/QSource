"""Three stores and three queries: full recompute, incremental exact, snapshot."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from qsource.const import (
    CLIP_BUYER,
    DELTA,
    FIXED_MU0,
    FIXED_TAU0_2,
    PKT_ROWS,
    SRC_BACKED,
    SRC_TRIAL,
    TAU0_FLOOR,
    W_MAX,
    Z_LB,
)
from qsource.store import hyper_from_backed as _hyper_from_backed

SEG = 100
TOP_K = 10
Q2_TH = 0.9
REFRESH_EVERY = 1000
SEED = [20261030, 2, 2]
TARGET_RECS_PER_SELLER = 20.0
HIGH_DENSITY_MIN_BUYERS = 5000
LOG_QUERY_TIMEOUT_S = 60.0
SRC_CODE_BACKED = 0
SRC_CODE_TRIAL = 1
CLIP_LO, CLIP_HI = CLIP_BUYER

def _arr_bytes(a) -> int:
    return int(sys.getsizeof(a)) + int(getattr(a, "nbytes", 0))


def rids_to_offers(rids) -> tuple[tuple[int, int], ...]:
    by: dict[int, int] = {}
    for item in rids or ():
        oid, row = int(item[0]), int(item[1])
        if row < 0:
            continue
        by[oid] = by.get(oid, 0) | (1 << row)
    return tuple(sorted(by.items()))


def parse_rids_json(s) -> tuple:
    if s is None or (isinstance(s, float) and not np.isfinite(s)) or s == "":
        return ()
    if not isinstance(s, str):
        s = str(s)
    return tuple((int(a), int(b)) for a, b in json.loads(s))


def n_buyers_for(n_sellers: int, recs_per_buyer: float) -> int:
    """Main grid: ~20 records per seller. N=10³ → ~1000 buyers."""
    return int(np.ceil(TARGET_RECS_PER_SELLER * n_sellers / max(float(recs_per_buyer), 1e-9)))


def n_buyers_high_density(n_sellers: int, recs_per_buyer: float) -> int:
    """Labeled extra cell, not mixed with the main grid. N=10³ → 5000 buyers."""
    return max(int(HIGH_DENSITY_MIN_BUYERS), n_buyers_for(n_sellers, recs_per_buyer))


def checkpoint_indices(n_buyers: int, n_ckpt: int) -> list[int]:
    n_buyers = int(n_buyers)
    n_ckpt = max(int(n_ckpt), 1)
    if n_buyers <= 0:
        return []
    if n_ckpt == 1:
        return [n_buyers - 1]
    out = [int(round(i * (n_buyers - 1) / (n_ckpt - 1))) for i in range(n_ckpt)]
    return sorted(set(out))


class Theta:
    """Incremental θ. Same branches as hyper_from_backed (prior='eb').
    θ uses packet-backed records only."""

    __slots__ = ("sse", "df", "sum_x", "sum_x2", "sum_invn", "K")

    def __init__(self):
        self.sse = 0.0
        self.df = 0.0
        self.sum_x = 0.0
        self.sum_x2 = 0.0
        self.sum_invn = 0.0
        self.K = 0

    def sub(self, n, s1, s2):
        n = int(n)
        if n <= 0:
            return
        x = float(s1) / n
        self.K -= 1
        self.sum_x -= x
        self.sum_x2 -= x * x
        self.sum_invn -= 1.0 / n
        if n >= 2:
            self.sse -= float(s2) - float(s1) ** 2 / n
            self.df -= n - 1

    def add(self, n, s1, s2):
        n = int(n)
        if n <= 0:
            return
        x = float(s1) / n
        self.K += 1
        self.sum_x += x
        self.sum_x2 += x * x
        self.sum_invn += 1.0 / n
        if n >= 2:
            self.sse += float(s2) - float(s1) ** 2 / n
            self.df += n - 1

    def hyper(self):
        k = int(self.K)
        if k < 2:
            mu0, tau0_2 = FIXED_MU0, FIXED_TAU0_2
        else:
            mu0 = float(self.sum_x / k)
            tau0_2 = None
        if self.df <= 0:
            if k >= 2:
                var_m = float((self.sum_x2 - self.sum_x ** 2 / k) / max(k - 1, 1))
                tau0_2 = max(var_m, TAU0_FLOOR)
                sigma2 = tau0_2
            else:
                mu0, tau0_2, sigma2 = FIXED_MU0, FIXED_TAU0_2, FIXED_TAU0_2
            return float(sigma2), float(mu0), float(tau0_2)
        sigma2 = max(float(self.sse / self.df), 1e-8)
        if k < 2:
            return float(sigma2), float(FIXED_MU0), float(FIXED_TAU0_2)
        var_m = float((self.sum_x2 - self.sum_x ** 2 / k) / max(k - 1, 1))
        mean_se = float(sigma2 * self.sum_invn / k)
        tau0_2 = var_m - mean_se
        if tau0_2 < TAU0_FLOOR:
            tau0_2 = TAU0_FLOOR
        return float(sigma2), float(mu0), float(tau0_2)

    def nbytes(self) -> int:
        return int(sys.getsizeof(self))


def hyper_from_arrays(n_g, s1_g, s2_g):
    n, s1, s2 = {}, {}, {}
    for s in np.flatnonzero(n_g >= 1).tolist():
        s = int(s)
        n[s] = int(n_g[s])
        s1[s] = float(s1_g[s])
        s2[s] = float(s2_g[s])
    return _hyper_from_backed(n, s1, s2, "eb")


def publish_eb_vec(n_eff, s1_eff, sigma2, mu0, tau0_2):
    n_eff = np.asarray(n_eff, float)
    s1_eff = np.asarray(s1_eff, float)
    empty = n_eff <= 1e-15
    prec = 1.0 / tau0_2 + n_eff / sigma2
    mj = (mu0 / tau0_2 + s1_eff / sigma2) / prec
    vj = prec ** -0.5
    mj = np.where(empty, mu0, mj)
    vj = np.where(empty, float(tau0_2) ** 0.5, vj)
    return mj.astype(np.float64, copy=False), vj.astype(np.float64, copy=False)


def publish_arrays(n_g, s1_g, n_u, s1_u, e_u, sigma2, mu0, tau0_2,
                   kappa=W_MAX, pkt=PKT_ROWS, delta=DELTA):
    n_g = np.asarray(n_g)
    n_u = np.asarray(n_u)
    s1_g = np.asarray(s1_g, float)
    s1_u = np.asarray(s1_u, float)
    e_u = np.asarray(e_u, float)
    n_g_f = n_g.astype(np.float64, copy=False)
    n_u_f = n_u.astype(np.float64, copy=False)
    wu = np.minimum(float(kappa), e_u / float(pkt))
    m_g, v_g = publish_eb_vec(n_g_f, s1_g, sigma2, mu0, tau0_2)
    m_unw, v_unw = publish_eb_vec(n_g_f + n_u_f, s1_g + s1_u, sigma2, mu0, tau0_2)
    has_u = n_u > 0
    mean_u = np.divide(s1_u, n_u_f, out=np.zeros_like(s1_u), where=has_u)
    n_mix = n_g_f + np.where(has_u, wu, 0.0)
    s1_mix = s1_g + np.where(has_u, wu * mean_u, 0.0)
    m_mix, v_mix = publish_eb_vec(n_mix, s1_mix, sigma2, mu0, tau0_2)
    in_risk = (n_g >= 1) & (n_u >= 1) & ((m_g - m_unw) > float(delta))
    no_g_u = (n_g == 0) & (n_u >= 1)
    m = np.where(in_risk, m_g, np.where(no_g_u, m_unw, m_mix))
    v = np.where(in_risk, v_g, np.where(no_g_u, v_unw, v_mix))
    pub = np.clip(m, CLIP_LO, CLIP_HI)
    lb = m - float(Z_LB) * v
    w_u = np.where(n_u >= 1, wu, 0.0)
    return dict(
        m_g=m_g, v_g=v_g, m_unw=m_unw, v_unw=v_unw, m_mix=m_mix, v_mix=v_mix,
        in_risk=in_risk, m=m, v=v, pub=pub, lb=lb, w_u=w_u,
    )


def _top_k_ids(ids, scores, k=TOP_K):
    ids = np.asarray(ids)
    scores = np.asarray(scores, float)
    if ids.size == 0:
        return [], []
    order = np.lexsort((ids, -scores))
    take = order[: min(int(k), ids.size)]
    sid = ids[take].astype(int, copy=False)
    val = scores[take]
    return [int(x) for x in sid], [float(x) for x in val]


def q1_from_arrays(n_g, s1_g, n_u, s1_u, e_u, theta, lo, hi):
    """O(segment). θ is passed in, computed once by the caller (backed records only)."""
    lo, hi = int(lo), int(hi)
    sl = slice(lo, hi + 1)
    parts = publish_arrays(n_g[sl], s1_g[sl], n_u[sl], s1_u[sl], e_u[sl], *theta)
    ids = np.arange(lo, hi + 1, dtype=np.int32)
    sids, vals = _top_k_ids(ids, parts["pub"], TOP_K)
    return list(zip(sids, vals))


def q2_from_arrays(n_g, s1_g, n_u, s1_u, e_u, theta):
    """O(N). (m, v) of the published branch; ℓ = m − 0.8416 v."""
    parts = publish_arrays(n_g, s1_g, n_u, s1_u, e_u, *theta)
    hit = np.flatnonzero(parts["lb"] > Q2_TH)
    return [(int(s), float(parts["lb"][s])) for s in hit]


def q3_from_arrays(n_g, s1_g, n_u, s1_u, e_u, theta, cand):
    """O(|C|). Risk among candidates, Top-10 by π_G − π_unw."""
    if not cand:
        return []
    c = np.fromiter(cand, dtype=np.int32)
    parts = publish_arrays(n_g[c], s1_g[c], n_u[c], s1_u[c], e_u[c], *theta)
    risk = np.asarray(parts["in_risk"], bool)
    if not np.any(risk):
        return []
    c_r = c[risk]
    gap = np.asarray(parts["m_g"], float)[risk] - np.asarray(parts["m_unw"], float)[risk]
    sids, vals = _top_k_ids(c_r, gap, TOP_K)
    return list(zip(sids, vals))


def as_query_state(obj) -> dict:
    """Normalize IncExact or a Log rebuild dict to the shared query view."""
    if isinstance(obj, dict):
        th = obj["theta"]
        if not isinstance(th, tuple):
            th = tuple(float(x) for x in th)
        return dict(
            n_g=obj["n_g"], s1_g=obj["s1_g"], n_u=obj["n_u"], s1_u=obj["s1_u"],
            e_u=obj["e_u"], C=obj["C"], theta=th,
        )
    return dict(
        n_g=obj.n_g, s1_g=obj.s1_g, n_u=obj.n_u, s1_u=obj.s1_u,
        e_u=obj.e_u, C=obj.C, theta=obj.theta.hyper(),
    )


def query_q1(state, lo, hi):
    """Shared Q1. θ computed once in as_query_state (backed records only). O(segment)."""
    st = as_query_state(state)
    return q1_from_arrays(st["n_g"], st["s1_g"], st["n_u"], st["s1_u"], st["e_u"],
                          st["theta"], lo, hi)


def query_q2(state):
    """Shared Q2. θ computed once (backed records only). O(N)."""
    st = as_query_state(state)
    return q2_from_arrays(st["n_g"], st["s1_g"], st["n_u"], st["s1_u"], st["e_u"],
                          st["theta"])


def query_q3(state):
    """Shared Q3. θ computed once (backed records only). O(|C|)."""
    st = as_query_state(state)
    return q3_from_arrays(st["n_g"], st["s1_g"], st["n_u"], st["s1_u"], st["e_u"],
                          st["theta"], st["C"])


def _or_bits(store, sid: int, offers, src_u: bool):
    table = store.u_bits if src_u else store.g_bits
    e = store.e_u if src_u else store.e_g
    d = table[sid]
    added = 0
    for oid, mask in offers:
        old = d.get(oid, 0)
        new = old | int(mask)
        if new != old:
            d[oid] = new
            added += (new ^ old).bit_count()
    if added:
        e[sid] += added


@dataclass
class Stream:
    n_sellers: int
    n_buyers: int
    n_recs: int
    seller: np.ndarray
    source: np.ndarray
    pi: np.ndarray
    offer_i: np.ndarray
    start: np.ndarray
    offers_pool: list
    n_rows: np.ndarray
    g_frac: float = 0.0
    recs_per_buyer: float = 0.0
    sellers_per_buyer: float = 0.0
    rows_per_buyer: float = 0.0
    n_g: int = 0
    n_u: int = 0
    template: dict = field(default_factory=dict)

    def buyer_slice(self, b: int):
        a, z = int(self.start[b]), int(self.start[b + 1])
        return a, z

    def nbytes(self, n_buyers=None) -> int:
        nb = self.n_buyers if n_buyers is None else int(n_buyers)
        end = int(self.start[nb]) if nb < len(self.start) else int(self.n_recs)
        n = 0
        for arr in (self.seller, self.source, self.pi, self.offer_i, self.n_rows):
            n += int(arr[:end].nbytes)
        n += int(self.start[: nb + 1].nbytes)
        for off in self.offers_pool:
            n += sys.getsizeof(off)
            for oid, mask in off:
                n += sys.getsizeof(oid) + sys.getsizeof(mask)
        return n


def _bitmap_nbytes(bits_list) -> int:
    n = sys.getsizeof(bits_list)
    for d in bits_list:
        n += sys.getsizeof(d)
        for k, v in d.items():
            n += sys.getsizeof(k) + sys.getsizeof(v)
    return n


class IncExact:
    """Per-seller columns + (seller, offer) backed/trial bitmaps + θ accumulators + C."""

    def __init__(self, n_sellers: int):
        n = int(n_sellers)
        self.n_sellers = n
        self.n_g = np.zeros(n, np.int32)
        self.s1_g = np.zeros(n, np.float64)
        self.s2_g = np.zeros(n, np.float64)
        self.n_u = np.zeros(n, np.int32)
        self.s1_u = np.zeros(n, np.float64)
        self.e_g = np.zeros(n, np.int32)
        self.e_u = np.zeros(n, np.int32)
        self.g_bits = [{} for _ in range(n)]
        self.u_bits = [{} for _ in range(n)]
        self.C: set[int] = set()
        self.theta = Theta()
        self.n_buyers = 0
        self.n_recs = 0

    def _touch_c(self, s: int):
        if self.n_g[s] >= 1 and self.n_u[s] >= 1:
            self.C.add(s)
        else:
            self.C.discard(s)

    def _write_rec(self, s: int, src: int, pi: float, offers):
        if src == SRC_CODE_BACKED:
            self.theta.sub(self.n_g[s], self.s1_g[s], self.s2_g[s])
            self.n_g[s] += 1
            self.s1_g[s] += pi
            self.s2_g[s] += pi * pi
            self.theta.add(self.n_g[s], self.s1_g[s], self.s2_g[s])
            _or_bits(self, s, offers, src_u=False)
        else:
            self.n_u[s] += 1
            self.s1_u[s] += pi
            _or_bits(self, s, offers, src_u=True)
        self._touch_c(s)
        self.n_recs += 1

    def write_buyer(self, stream: Stream, b: int) -> float:
        t0 = time.perf_counter()
        a, z = stream.buyer_slice(b)
        pool = stream.offers_pool
        sl_s = stream.seller
        sl_src = stream.source
        sl_pi = stream.pi
        sl_oi = stream.offer_i
        for i in range(a, z):
            self._write_rec(int(sl_s[i]), int(sl_src[i]), float(sl_pi[i]), pool[int(sl_oi[i])])
        self.n_buyers += 1
        return time.perf_counter() - t0

    def theta_tuple(self):
        return self.theta.hyper()

    def q1(self, lo, hi):
        t0 = time.perf_counter()
        rows = query_q1(self, lo, hi)
        return time.perf_counter() - t0, rows

    def q2(self):
        t0 = time.perf_counter()
        rows = query_q2(self)
        return time.perf_counter() - t0, rows

    def q3(self):
        t0 = time.perf_counter()
        rows = query_q3(self)
        return time.perf_counter() - t0, rows

    def nbytes_parts(self) -> dict:
        obj = int(sys.getsizeof(self) + sys.getsizeof(getattr(self, "__dict__", {})))
        seller = obj + sum(
            _arr_bytes(x) for x in
            (self.n_g, self.s1_g, self.s2_g, self.n_u, self.s1_u, self.e_g, self.e_u)
        )
        bits = _bitmap_nbytes(self.g_bits) + _bitmap_nbytes(self.u_bits)
        acc = int(self.theta.nbytes())
        cand = sys.getsizeof(self.C) + sum(sys.getsizeof(x) for x in self.C)
        return dict(
            seller_mb=seller / (1024 * 1024),
            bitmap_mb=bits / (1024 * 1024),
            acc_mb=acc / (1024 * 1024),
            cand_mb=cand / (1024 * 1024),
            log_mb=0.0,
            snapshot_mb=0.0,
            total_mb=(seller + bits + acc + cand) / (1024 * 1024),
            includes_python_object_overhead=True,
        )


def rebuild_from_arrays(n_sellers, seller, source, pi, offer_i, n_recs, offers_pool):
    """One-pass Full Recompute rebuild. O(records + N), not O(N × records)."""
    n = int(n_sellers)
    n_g = np.zeros(n, np.int32)
    s1_g = np.zeros(n, np.float64)
    s2_g = np.zeros(n, np.float64)
    n_u = np.zeros(n, np.int32)
    s1_u = np.zeros(n, np.float64)
    e_g = np.zeros(n, np.int32)
    e_u = np.zeros(n, np.int32)
    g_bits = [{} for _ in range(n)]
    u_bits = [{} for _ in range(n)]
    dummy = type("B", (), {"g_bits": g_bits, "u_bits": u_bits, "e_g": e_g, "e_u": e_u})()
    end = int(n_recs)
    pool = offers_pool
    for i in range(end):
        s = int(seller[i])
        src = int(source[i])
        ph = float(pi[i])
        offers = pool[int(offer_i[i])]
        if src == SRC_CODE_BACKED:
            n_g[s] += 1
            s1_g[s] += ph
            s2_g[s] += ph * ph
            _or_bits(dummy, s, offers, src_u=False)
        else:
            n_u[s] += 1
            s1_u[s] += ph
            _or_bits(dummy, s, offers, src_u=True)
    C = set(int(s) for s in np.flatnonzero((n_g >= 1) & (n_u >= 1)).tolist())
    th = hyper_from_arrays(n_g, s1_g, s2_g)
    return dict(n_g=n_g, s1_g=s1_g, s2_g=s2_g, n_u=n_u, s1_u=s1_u,
                e_g=e_g, e_u=e_u, C=C, theta=th)


def rebuild_from_stream(stream: Stream, n_buyers: int):
    end = int(stream.start[n_buyers])
    return rebuild_from_arrays(
        stream.n_sellers, stream.seller, stream.source, stream.pi,
        stream.offer_i, end, stream.offers_pool,
    )


class LogStore:
    def __init__(self, stream: Stream):
        self.stream = stream
        n = int(stream.n_recs)
        self.seller = np.empty(n, np.int32)
        self.source = np.empty(n, np.uint8)
        self.pi = np.empty(n, np.float64)
        self.offer_i = np.empty(n, np.int32)
        self.start = np.empty(int(stream.n_buyers) + 1, np.int32)
        self.start[0] = 0
        self.n_buyers = 0
        self.n_recs = 0
        self._last_rebuild_s = 0.0

    def write_buyer(self, stream: Stream, b: int) -> float:
        t0 = time.perf_counter()
        a, z = stream.buyer_slice(b)
        p = self.n_recs
        k = z - a
        self.seller[p:p + k] = stream.seller[a:z]
        self.source[p:p + k] = stream.source[a:z]
        self.pi[p:p + k] = stream.pi[a:z]
        self.offer_i[p:p + k] = stream.offer_i[a:z]
        self.n_recs = p + k
        self.n_buyers += 1
        self.start[self.n_buyers] = self.n_recs
        return time.perf_counter() - t0

    def rebuild(self):
        t0 = time.perf_counter()
        st = rebuild_from_arrays(
            self.stream.n_sellers, self.seller, self.source, self.pi,
            self.offer_i, self.n_recs, self.stream.offers_pool,
        )
        self._last_rebuild_s = time.perf_counter() - t0
        return st

    def _query(self, fn, timeout=LOG_QUERY_TIMEOUT_S):
        t0 = time.perf_counter()
        st = self.rebuild()
        if time.perf_counter() - t0 > timeout:
            return time.perf_counter() - t0, None, True
        rows = fn(st)
        dt = time.perf_counter() - t0
        return dt, rows, dt > timeout

    def q1(self, lo, hi):
        return self._query(lambda st: query_q1(st, lo, hi))

    def q2(self):
        return self._query(query_q2)

    def q3(self):
        return self._query(query_q3)

    def nbytes_parts(self) -> dict:
        end = int(self.n_recs)
        obj = int(sys.getsizeof(self) + sys.getsizeof(getattr(self, "__dict__", {})))
        log = obj + sum(_arr_bytes(a) for a in (
            self.seller[:end], self.source[:end], self.pi[:end],
            self.offer_i[:end], self.start[: self.n_buyers + 1],
        ))
        log += sys.getsizeof(self.stream.offers_pool)
        for off in self.stream.offers_pool:
            log += sys.getsizeof(off)
            for oid, mask in off:
                log += sys.getsizeof(oid) + sys.getsizeof(mask)
        return dict(
            seller_mb=0.0, bitmap_mb=0.0, acc_mb=0.0, cand_mb=0.0, snapshot_mb=0.0,
            log_mb=log / (1024 * 1024),
            total_mb=log / (1024 * 1024),
            includes_python_object_overhead=True,
        )


class IncSnapshot(IncExact):
    """Same maintenance as Incremental Exact; queries read materialised columns. Approximate."""

    def __init__(self, n_sellers: int):
        super().__init__(n_sellers)
        n = int(n_sellers)
        self.mat_pub = np.full(n, np.nan, np.float64)
        self.mat_lb = np.full(n, np.nan, np.float64)
        self.mat_risk = np.zeros(n, np.int8)
        self.mat_pi_g = np.full(n, np.nan, np.float64)
        self.mat_pi_equal = np.full(n, np.nan, np.float64)
        self.refresh_s: list[float] = []
        self.n_refresh = 0
        self._fresh = False

    def refresh(self) -> float:
        t0 = time.perf_counter()
        th = self.theta.hyper()
        parts = publish_arrays(self.n_g, self.s1_g, self.n_u, self.s1_u, self.e_u, *th)
        self.mat_pub = parts["pub"]
        self.mat_lb = parts["lb"]
        self.mat_risk = parts["in_risk"].astype(np.int8, copy=False)
        self.mat_pi_g = parts["m_g"]
        self.mat_pi_equal = parts["m_unw"]
        self._fresh = True
        dt = time.perf_counter() - t0
        self.refresh_s.append(dt)
        self.n_refresh += 1
        return dt

    def maybe_refresh(self) -> float:
        if self.n_buyers > 0 and self.n_buyers % REFRESH_EVERY == 0:
            return self.refresh()
        return 0.0

    def q1(self, lo, hi):
        """Read mat_pub only. Does not call query_q1 or IncExact.q1."""
        t0 = time.perf_counter()
        lo, hi = int(lo), int(hi)
        ids = np.arange(lo, hi + 1, dtype=np.int32)
        sids, vals = _top_k_ids(ids, self.mat_pub[lo: hi + 1], TOP_K)
        return time.perf_counter() - t0, list(zip(sids, vals))

    def q2(self):
        """Read mat_lb only. Does not call query_q2 or IncExact.q2."""
        t0 = time.perf_counter()
        hit = np.flatnonzero(self.mat_lb > Q2_TH)
        rows = [(int(s), float(self.mat_lb[s])) for s in hit]
        return time.perf_counter() - t0, rows

    def q3(self):
        """Read mat_risk / mat_pi_g / mat_pi_equal only. Does not call query_q3 or IncExact.q3."""
        t0 = time.perf_counter()
        hit = np.flatnonzero(self.mat_risk.astype(bool))
        if hit.size == 0:
            return time.perf_counter() - t0, []
        gap = self.mat_pi_g[hit] - self.mat_pi_equal[hit]
        sids, vals = _top_k_ids(hit, gap, TOP_K)
        return time.perf_counter() - t0, list(zip(sids, vals))

    def nbytes_parts(self) -> dict:
        base = super().nbytes_parts()
        extra = sum(_arr_bytes(x) for x in
                    (self.mat_pub, self.mat_lb, self.mat_risk, self.mat_pi_g, self.mat_pi_equal))
        extra += sys.getsizeof(self.refresh_s) + sum(sys.getsizeof(x) for x in self.refresh_s)
        base["snapshot_mb"] = extra / (1024 * 1024)
        base["total_mb"] = float(base["total_mb"]) + extra / (1024 * 1024)
        base["includes_python_object_overhead"] = True
        return base


_TEMPLATE_CACHE = None


def load_template_groups() -> dict:
    """No silent synthetic default. Callers pass a template or use --synthetic."""
    if _TEMPLATE_CACHE is None:
        raise FileNotFoundError(
            "no record template loaded; pass records from config=none forced_trials "
            "or call synthetic_template() explicitly"
        )
    return _TEMPLATE_CACHE


def map_sellers(orig, n_sellers: int, rng: np.random.Generator) -> dict[int, int]:
    orig = list(dict.fromkeys(int(s) for s in orig))
    n_sellers = int(n_sellers)
    if not orig:
        return {}
    n_seg = max(n_sellers // SEG, 1)
    k = len(orig)
    if n_sellers >= SEG and k <= SEG:
        base = int(rng.integers(0, n_seg)) * SEG
        seg_len = min(SEG, n_sellers - base)
        if k <= seg_len:
            slots = rng.choice(seg_len, size=k, replace=False)
            return {s: int(base + slots[i]) for i, s in enumerate(orig)}
    if k <= n_sellers:
        chosen = rng.choice(n_sellers, size=k, replace=False)
        return {s: int(chosen[i]) for i, s in enumerate(orig)}
    return {s: int(rng.integers(0, n_sellers)) for s in orig}


def make_stream(n_sellers: int, n_buyers: int, rng: np.random.Generator,
                template: dict | None = None) -> Stream:
    tpl = template if template is not None else load_template_groups()
    groups = tpl["groups"]
    pool = tpl["offers_pool"]
    n_g = len(groups)
    n_buyers = int(n_buyers)
    n_sellers = int(n_sellers)
    recs_est = int(sum(len(groups[int(rng.integers(0, n_g))]) for _ in range(min(n_buyers, 8)))
                   / max(min(n_buyers, 8), 1) * n_buyers) if n_g else 0
    cap = max(recs_est + 16, n_buyers * 4)
    seller = np.empty(cap, np.int32)
    source = np.empty(cap, np.uint8)
    pi = np.empty(cap, np.float64)
    offer_i = np.empty(cap, np.int32)
    n_rows_a = np.empty(cap, np.int32)
    start = np.empty(n_buyers + 1, np.int32)
    start[0] = 0
    ptr = 0
    n_g_rec = n_u_rec = 0
    sellers_per = []
    rows_per = []
    for b in range(n_buyers):
        g = groups[int(rng.integers(0, n_g))]
        mp = map_sellers((t[0] for t in g), n_sellers, rng)
        need = ptr + len(g)
        if need > cap:
            cap = int(max(need, cap * 2))
            seller = np.resize(seller, cap)
            source = np.resize(source, cap)
            pi = np.resize(pi, cap)
            offer_i = np.resize(offer_i, cap)
            n_rows_a = np.resize(n_rows_a, cap)
        sids = []
        rows_b = 0
        for orig_s, src, ph, oi, nr in g:
            seller[ptr] = mp[int(orig_s)]
            source[ptr] = src
            pi[ptr] = ph
            offer_i[ptr] = oi
            n_rows_a[ptr] = nr
            sids.append(int(seller[ptr]))
            rows_b += int(nr)
            if src == SRC_CODE_BACKED:
                n_g_rec += 1
            else:
                n_u_rec += 1
            ptr += 1
        start[b + 1] = ptr
        sellers_per.append(len(set(sids)))
        rows_per.append(rows_b)
    n_recs = ptr
    recs_per = float(n_recs / max(n_buyers, 1))
    return Stream(
        n_sellers=n_sellers, n_buyers=n_buyers, n_recs=n_recs,
        seller=seller[:n_recs].copy(), source=source[:n_recs].copy(),
        pi=pi[:n_recs].copy(), offer_i=offer_i[:n_recs].copy(),
        start=start, offers_pool=pool, n_rows=n_rows_a[:n_recs].copy(),
        g_frac=float(n_g_rec / max(n_recs, 1)),
        recs_per_buyer=recs_per,
        sellers_per_buyer=float(np.mean(sellers_per)) if sellers_per else 0.0,
        rows_per_buyer=float(np.mean(rows_per)) if rows_per else 0.0,
        n_g=n_g_rec, n_u=n_u_rec,
        template=dict(n_files=tpl["n_files"], n_groups=tpl["n_groups"],
                      recs_per_buyer_tpl=tpl["recs_per_buyer"],
                      sellers_per_buyer_tpl=tpl["sellers_per_buyer"],
                      g_frac_tpl=tpl["g_frac"],
                      rows_per_rec_tpl=tpl["rows_per_rec"],
                      max_rows_per_rec=tpl["max_rows_per_rec"]),
    )

def n_buyers_for(n_sellers: int, recs_per_buyer: float) -> int:
    return int(np.ceil(TARGET_RECS_PER_SELLER * n_sellers / max(float(recs_per_buyer), 1e-9)))


def synthetic_template(n_groups: int = 200, recs_per: int = 8, rng=None) -> dict:
    rng = np.random.default_rng(0 if rng is None else rng)
    offers_pool = []
    groups = []
    n_g = n_u = 0
    n_rows_all = []
    recs_per_g = []
    sellers_per = []
    for _ in range(int(n_groups)):
        recs = []
        n_rec = int(recs_per)
        orig = [int(x) for x in rng.integers(0, 8, size=n_rec)]
        for s in orig:
            n_rows = int(rng.choice([10, 20, 40]))
            mask = (1 << n_rows) - 1
            oid = int(rng.integers(0, 10_000))
            offers = ((oid, mask),)
            oi = len(offers_pool)
            offers_pool.append(offers)
            src = SRC_CODE_BACKED if float(rng.random()) < 0.55 else SRC_CODE_TRIAL
            if src == SRC_CODE_BACKED:
                n_g += 1
            else:
                n_u += 1
            pi = float(rng.uniform(0.2, 1.0))
            n_rows_all.append(n_rows)
            recs.append((int(s), src, pi, oi, n_rows))
        groups.append(recs)
        recs_per_g.append(len(recs))
        sellers_per.append(len({t[0] for t in recs}))
    return dict(
        groups=groups, offers_pool=offers_pool, n_files=0,
        n_groups=len(groups), n_template_recs=n_g + n_u,
        n_g=n_g, n_u=n_u, g_frac=float(n_g / max(n_g + n_u, 1)),
        recs_per_buyer=float(np.mean(recs_per_g)) if recs_per_g else 0.0,
        sellers_per_buyer=float(np.mean(sellers_per)) if sellers_per else 0.0,
        rows_per_rec=float(np.mean(n_rows_all)) if n_rows_all else 0.0,
        rows_per_buyer=float(np.mean([sum(t[4] for t in g) for g in groups])) if groups else 0.0,
        max_rows_per_rec=int(max(n_rows_all) if n_rows_all else 0),
    )


def template_from_records(files, config=None, method=None, designs=None, policies=None) -> dict:
    """Build a stream template from records CSVs (seller, source, pi_hat, rids_json)."""
    offers_pool: list = []
    offer_key = {}
    groups = []
    n_g = n_u = 0
    n_rows_all = []
    recs_per = []
    sellers_per = []
    files = list(files)
    for p in files:
        df = pd.read_csv(p)
        if config is not None and "config" in df.columns:
            df = df[df["config"].astype(str) == str(config)]
        if method is not None and "method" in df.columns:
            df = df[df["method"].astype(str) == str(method)]
        if designs is not None and "design" in df.columns:
            df = df[df["design"].astype(str).isin(list(designs))]
        if policies is not None and "policy" in df.columns:
            df = df[df["policy"].astype(str).isin(list(policies))]
        if not len(df):
            continue
        if all(c in df.columns for c in ("dataset", "instance", "policy", "buyer_t", "is_writer")):
            keys = ["dataset", "instance", "policy", "buyer_t", "is_writer"]
        else:
            keys = [c for c in (
                "dataset", "instance", "design", "method", "buyer", "is_attacker",
            ) if c in df.columns]
            if "buyer" not in keys and "buyer_t" in df.columns:
                keys.append("buyer_t")
        if not keys:
            keys = [c for c in df.columns if c in ("buyer_t", "buyer", "policy")]
        for _, sl in df.groupby(keys, sort=False):
            recs = []
            for r in sl.itertuples(index=False):
                raw = getattr(r, "rids_json", "[]")
                rids = parse_rids_json(raw)
                offers = rids_to_offers(rids)
                if offers not in offer_key:
                    offer_key[offers] = len(offers_pool)
                    offers_pool.append(offers)
                oi = offer_key[offers]
                src_s = str(getattr(r, "source", "packet_backed"))
                src = SRC_CODE_BACKED if src_s in (
                    "packet_backed", "platform_checked", "backed",
                    "packet-backed", "platform-checked",
                ) else SRC_CODE_TRIAL
                if src == SRC_CODE_BACKED:
                    n_g += 1
                else:
                    n_u += 1
                n_rows = sum(m.bit_count() for _, m in offers)
                n_rows_all.append(n_rows)
                pi = float(getattr(r, "pi_hat", getattr(r, "pi_hat_jb", 0.5)))
                recs.append((int(r.seller), src, pi, oi, n_rows))
            if recs:
                groups.append(recs)
                recs_per.append(len(recs))
                sellers_per.append(len({t[0] for t in recs}))
    return dict(
        groups=groups, offers_pool=offers_pool, n_files=len(list(files)),
        n_groups=len(groups), n_template_recs=n_g + n_u,
        n_g=n_g, n_u=n_u, g_frac=float(n_g / max(n_g + n_u, 1)),
        recs_per_buyer=float(np.mean(recs_per)) if recs_per else 0.0,
        sellers_per_buyer=float(np.mean(sellers_per)) if sellers_per else 0.0,
        rows_per_rec=float(np.mean(n_rows_all)) if n_rows_all else 0.0,
        rows_per_buyer=float(np.mean([sum(t[4] for t in g) for g in groups])) if groups else 0.0,
        max_rows_per_rec=int(max(n_rows_all) if n_rows_all else 0),
    )


NUM_TARGET = 1e-12
NUM_PASS = 1e-9


def _max_abs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def compare_exact_vs_log(inc: IncExact, log_st: dict) -> dict:
    n = inc.n_sellers
    th_i = inc.theta.hyper()
    th_l = tuple(float(x) for x in log_st["theta"])
    theta_err = [abs(a - b) for a, b in zip(th_i, th_l)]
    fields_int = ("n_g", "n_u", "e_g", "e_u")
    fields_f = ("s1_g", "s2_g", "s1_u")
    int_mis = {}
    for k in fields_int:
        d = int(np.max(np.abs(inc.__dict__[k] - log_st[k]))) if n else 0
        int_mis[k] = d
    f_err = {k: _max_abs(inc.__dict__[k], log_st[k]) for k in fields_f}
    parts_i = publish_arrays(inc.n_g, inc.s1_g, inc.n_u, inc.s1_u, inc.e_u, *th_i)
    parts_l = publish_arrays(
        log_st["n_g"], log_st["s1_g"], log_st["n_u"], log_st["s1_u"], log_st["e_u"], *th_l,
    )
    pub_keys = ("m_g", "m_mix", "m_unw", "pub", "lb", "w_u")
    pub_err = {k: _max_abs(parts_i[k], parts_l[k]) for k in pub_keys}
    risk_mis = int(np.sum(parts_i["in_risk"] != parts_l["in_risk"]))
    c_mis = int(len(inc.C ^ log_st["C"]))

    def q1_all(state):
        rows = []
        for lo in range(0, n, SEG):
            hi = min(lo + SEG - 1, n - 1)
            rows.extend((lo, s, v) for s, v in query_q1(state, lo, hi))
        return rows

    def q_err(a, b, keyed=False):
        if keyed:
            sa, sb = [(x[0], x[1]) for x in a], [(x[0], x[1]) for x in b]
            va, vb = [x[2] for x in a], [x[2] for x in b]
        else:
            sa, sb = [x[0] for x in a], [x[0] for x in b]
            va, vb = [x[1] for x in a], [x[1] for x in b]
        set_mis = int(len(set(sa) ^ set(sb)))
        order_mis = int(sa != sb)
        val_err = 0.0
        if sa == sb and va:
            val_err = max(abs(x - y) for x, y in zip(va, vb))
        elif va and vb and len(va) == len(vb):
            val_err = max(abs(x - y) for x, y in zip(va, vb))
        return dict(set_mismatch=set_mis, order_mismatch=order_mis, val_err=float(val_err))

    qerr = dict(
        q1=q_err(q1_all(inc), q1_all(log_st), keyed=True),
        q2=q_err(query_q2(inc), query_q2(log_st)),
        q3=q_err(query_q3(inc), query_q3(log_st)),
    )
    max_num = max(
        [max(theta_err), max(f_err.values()), max(pub_err.values())]
        + [qerr[q]["val_err"] for q in qerr]
    )
    int_bad = sum(int_mis.values()) + risk_mis + c_mis
    q_set_bad = sum(qerr[q]["set_mismatch"] + qerr[q]["order_mismatch"] for q in qerr)
    ok = int_bad == 0 and q_set_bad == 0 and max_num < NUM_PASS
    return dict(
        ok=bool(ok),
        max_numeric=float(max_num),
        int_mismatch=int(int_bad),
        query_mismatch=int(q_set_bad),
        theta_err=theta_err,
        queries=qerr,
    )


FullRecompute = LogStore
IncrementalExact = IncExact
IncrementalSnapshot = IncSnapshot
