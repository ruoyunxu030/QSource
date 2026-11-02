"""Shared stores: row-level Beta, buyer-level empirical Bayes, and QSource."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qsource.const import (
    BACKED_SOURCES,
    BETA_A,
    BETA_B,
    CLIP_BUYER,
    CLIP_ROW,
    FIXED_MU0,
    FIXED_TAU0_2,
    KIND_BUYER,
    KIND_ROW,
    PKT_ROWS,
    SRC_BACKED,
    SRC_TRIAL,
    TAU0_FLOOR,
    W_MAX,
    DELTA,
)


def hyper_from_backed(n: dict, s1: dict, s2: dict, prior: str = "eb"):
    js = [s for s, nj in n.items() if nj >= 1]
    js2 = [s for s in js if n[s] >= 2]
    if prior == "fixed" or len(js) < 2:
        mu0, tau0_2 = FIXED_MU0, FIXED_TAU0_2
    else:
        means = [s1[s] / n[s] for s in js]
        mu0 = float(sum(means) / len(means))
        tau0_2 = None
    if not js2:
        if prior == "eb" and len(js) >= 2:
            means = [s1[s] / n[s] for s in js]
            var_m = float(sum((m - mu0) ** 2 for m in means) / max(len(means) - 1, 1))
            tau0_2 = max(var_m, TAU0_FLOOR)
            sigma2 = tau0_2
        else:
            mu0, tau0_2, sigma2 = FIXED_MU0, FIXED_TAU0_2, FIXED_TAU0_2
        return float(sigma2), float(mu0), float(tau0_2)
    sse = 0.0
    df = 0
    for s in js2:
        nj = n[s]
        sse += s2[s] - s1[s] ** 2 / nj
        df += nj - 1
    sigma2 = max(float(sse / max(df, 1)), 1e-8)
    if prior == "fixed" or len(js) < 2:
        return float(sigma2), float(FIXED_MU0), float(FIXED_TAU0_2)
    means = [s1[s] / n[s] for s in js]
    mu0 = float(sum(means) / len(means))
    var_m = float(sum((m - mu0) ** 2 for m in means) / max(len(means) - 1, 1))
    mean_se = float(sum(sigma2 / n[s] for s in js) / len(js))
    tau0_2 = var_m - mean_se
    if tau0_2 < TAU0_FLOOR:
        tau0_2 = TAU0_FLOOR
    return float(sigma2), float(mu0), float(tau0_2)


def publish_eb(n_eff: float, s1_eff: float, sigma2: float, mu0: float, tau0_2: float):
    if n_eff <= 1e-15:
        return float(mu0), float(tau0_2 ** 0.5)
    prec = 1.0 / tau0_2 + float(n_eff) / sigma2
    mj = (mu0 / tau0_2 + float(s1_eff) / sigma2) / prec
    vj = prec ** -0.5
    return float(mj), float(vj)


def beta_mean(weights: list[tuple[float, float]]) -> float:
    if not weights:
        return float(BETA_A / (BETA_A + BETA_B))
    a = float(BETA_A)
    b = float(BETA_B)
    for g, w in weights:
        a += float(w) * float(g)
        b += float(w) * (1.0 - float(g))
    return float(a / (a + b))


@dataclass
class EvidenceStore:
    """Row-level store: each bought row once, seller posterior Beta(2,1)."""

    seller_of_offer: dict[int, int]
    row_sum: dict[tuple[int, int], float] = field(default_factory=dict)
    row_n: dict[tuple[int, int], int] = field(default_factory=dict)

    def update(self, gamma_by_row: dict[tuple[int, int], float]) -> None:
        for rid, g in gamma_by_row.items():
            self.row_sum[rid] = self.row_sum.get(rid, 0.0) + float(g)
            self.row_n[rid] = self.row_n.get(rid, 0) + 1

    def row_mean(self, rid: tuple[int, int]) -> float:
        return self.row_sum[rid] / max(self.row_n[rid], 1)

    def rows_of_seller(self, sid: int) -> list[tuple[int, int]]:
        sid = int(sid)
        return [rid for rid in self.row_sum if self.seller_of_offer.get(rid[0]) == sid]

    def n_distinct(self, sid: int) -> int:
        return len(self.rows_of_seller(sid))

    def seller_posterior(self, sid: int) -> tuple[float, float, float, int]:
        rids = self.rows_of_seller(sid)
        if not rids:
            a, b = BETA_A, BETA_B
            return a, b, a / (a + b), 0
        gbar = [self.row_mean(r) for r in rids]
        a = BETA_A + float(sum(gbar))
        b = BETA_B + float(sum(1.0 - g for g in gbar))
        return a, b, a / (a + b), len(rids)

    def beta_std(self, sid: int) -> float:
        a, b, _m, _n = self.seller_posterior(sid)
        den = (a + b) ** 2 * (a + b + 1.0)
        return float((a * b / den) ** 0.5) if den > 0 else 0.0


@dataclass
class BuyerEvidenceStore:
    """One record per (buyer, seller): π̂_jb = mean γ on rows bought from j."""

    n: dict[int, int] = field(default_factory=dict)
    s1: dict[int, float] = field(default_factory=dict)
    s2: dict[int, float] = field(default_factory=dict)
    nrows: dict[int, int] = field(default_factory=dict)
    prior: str = "eb"

    def n_buyers(self, sid: int) -> int:
        return int(self.n.get(int(sid), 0))

    def skip_score(self, sid: int) -> float:
        return float(self.n_buyers(sid))

    def n_rows(self, sid: int) -> int:
        return int(self.nrows.get(int(sid), 0))

    def update_buyer(self, sid: int, pi_hat: float, n_rows: int, t: int) -> None:
        sid = int(sid)
        p = float(pi_hat)
        self.n[sid] = self.n.get(sid, 0) + 1
        self.s1[sid] = self.s1.get(sid, 0.0) + p
        self.s2[sid] = self.s2.get(sid, 0.0) + p * p
        self.nrows[sid] = self.nrows.get(sid, 0) + int(n_rows)

    def _sigma2_tau(self) -> tuple[float, float, float]:
        js = [s for s, nj in self.n.items() if nj >= 1]
        js2 = [s for s in js if self.n[s] >= 2]
        if self.prior == "fixed" or len(js) < 2:
            mu0, tau0_2 = FIXED_MU0, FIXED_TAU0_2
        else:
            means = [self.s1[s] / self.n[s] for s in js]
            mu0 = float(sum(means) / len(means))
            tau0_2 = None
        if not js2:
            sigma2 = FIXED_TAU0_2 if self.prior == "fixed" or len(js) < 2 else FIXED_TAU0_2
            if self.prior == "eb" and len(js) >= 2:
                means = [self.s1[s] / self.n[s] for s in js]
                var_m = float(sum((m - mu0) ** 2 for m in means) / max(len(means) - 1, 1))
                tau0_2 = max(var_m, TAU0_FLOOR)
                sigma2 = tau0_2
            else:
                mu0, tau0_2, sigma2 = FIXED_MU0, FIXED_TAU0_2, FIXED_TAU0_2
            return sigma2, mu0, tau0_2
        sse = 0.0
        df = 0
        for s in js2:
            nj = self.n[s]
            sse += self.s2[s] - self.s1[s] ** 2 / nj
            df += nj - 1
        sigma2 = float(sse / max(df, 1))
        sigma2 = max(sigma2, 1e-8)
        if self.prior == "fixed" or len(js) < 2:
            return sigma2, FIXED_MU0, FIXED_TAU0_2
        means = [self.s1[s] / self.n[s] for s in js]
        mu0 = float(sum(means) / len(means))
        var_m = float(sum((m - mu0) ** 2 for m in means) / max(len(means) - 1, 1))
        mean_se = float(sum(sigma2 / self.n[s] for s in js) / len(js))
        tau0_2 = var_m - mean_se
        if tau0_2 < TAU0_FLOOR:
            tau0_2 = TAU0_FLOOR
        return sigma2, mu0, float(tau0_2)

    def publish(self, sid: int) -> tuple[float, float, int, float, float, float]:
        sid = int(sid)
        sigma2, mu0, tau0_2 = self._sigma2_tau()
        nj = self.n_buyers(sid)
        if nj <= 0:
            return float(mu0), float(tau0_2 ** 0.5), 0, mu0, tau0_2, sigma2
        prec = 1.0 / tau0_2 + nj / sigma2
        mj = (mu0 / tau0_2 + self.s1[sid] / sigma2) / prec
        vj = prec ** -0.5
        return float(mj), float(vj), nj, mu0, tau0_2, sigma2


@dataclass
class ArrivalSkipStore(BuyerEvidenceStore):
    """Skip score for skip-threshold > 1: number of distinct write times.

    A platform-checked record and a buyer record at the same t count as one.
    t≤0 (injected accounts) is ignored.
    """

    arrivals: dict = field(default_factory=dict)

    def update_buyer(self, sid: int, pi_hat: float, n_rows: int, t: int) -> None:
        super().update_buyer(sid, pi_hat, n_rows, t)
        if int(t) > 0:
            self.arrivals.setdefault(int(sid), set()).add(int(t))

    def skip_score(self, sid: int) -> float:
        return float(len(self.arrivals.get(int(sid), set())))


def _trim_stat(vals: list[float]) -> tuple[float, int]:
    xs = [float(x) for x in vals]
    n = len(xs)
    if n < 3:
        return (float(np.mean(xs)) if n else 0.0), n
    xs = sorted(xs)
    keep = xs[1:-1]
    return float(np.mean(keep)), int(len(keep))


@dataclass
class RobustBuyerStore(BuyerEvidenceStore):
    """Same writes and θ as equal-weight. median replaces the mean; n unchanged.
    trimmed_mean (agg='trim'): if n≥3 drop one min and one max, use remaining mean and n'.
    """

    agg: str = "mean"
    values: dict = field(default_factory=dict)

    def update_buyer(self, sid: int, pi_hat: float, n_rows: int, t: int) -> None:
        super().update_buyer(sid, pi_hat, n_rows, t)
        sid = int(sid)
        self.values.setdefault(sid, []).append(float(pi_hat))

    def publish(self, sid: int):
        sid = int(sid)
        if str(self.agg) == "mean":
            return super().publish(sid)
        sigma2, mu0, tau0_2 = self._sigma2_tau()
        vals = list(self.values.get(sid) or [])
        nj = int(len(vals))
        if nj <= 0:
            return float(mu0), float(tau0_2 ** 0.5), 0, mu0, tau0_2, sigma2
        if str(self.agg) == "median":
            loc = float(np.median(np.asarray(vals, float)))
            n_use = nj
        elif str(self.agg) == "trim":
            loc, n_use = _trim_stat(vals)
        else:
            raise ValueError(f"unknown agg {self.agg}")
        m, v = publish_eb(float(n_use), float(n_use) * float(loc), sigma2, mu0, tau0_2)
        return float(m), float(v), int(n_use), mu0, tau0_2, sigma2


@dataclass
class PurchaseWeightStore(BuyerEvidenceStore):
    """Purchase Weight. θ is the unweighted Equal Weight estimator.

    Packet-backed and platform-checked records have weight 1. Trial-only
    records have weight min(1, n_rows / m). Injection is a trial-only write, so s/m.
    Publish uses n_w = Σ w_i and loc = Σ w_i x_i / n_w. skip_score is Σ w_i.

    The third publish() value is not raw n_w. It is
    n_ret = round(n_w) if n_w ≥ 1 else 0. means_from_buyer_store skips
    sellers with n_ret ≤ 0 when building π_override, so this integer is a
    decision input, and the paper's results use this rule. skip_score
    stays the float Σ w used by the skip-threshold test.
    """

    pkt_rows: int = PKT_ROWS
    weights: dict = field(default_factory=dict)
    wvals: dict = field(default_factory=dict)

    def update_buyer(self, sid: int, pi_hat: float, n_rows: int, t: int,
                     weight: float | None = None) -> None:
        super().update_buyer(sid, pi_hat, n_rows, t)
        sid = int(sid)
        if weight is None:
            weight = min(1.0, float(n_rows) / float(self.pkt_rows))
        self.weights.setdefault(sid, []).append(float(weight))
        self.wvals.setdefault(sid, []).append(float(pi_hat))

    def skip_score(self, sid: int) -> float:
        return float(sum(self.weights.get(int(sid), []) or []))

    def publish(self, sid: int):
        sid = int(sid)
        sigma2, mu0, tau0_2 = self._sigma2_tau()
        ws = list(self.weights.get(sid) or [])
        vs = list(self.wvals.get(sid) or [])
        n_w = float(sum(ws))
        if n_w <= 0 or not vs:
            return float(mu0), float(tau0_2 ** 0.5), 0, mu0, tau0_2, sigma2
        loc = float(sum(w * v for w, v in zip(ws, vs)) / n_w)
        m, v = publish_eb(n_w, n_w * loc, sigma2, mu0, tau0_2)
        # Decision count for π_override (means_from_buyer_store: n<=0 omitted).
        # Not a display-only field. Do not return float n_w.
        n_ret = int(round(n_w)) if n_w >= 1.0 - 1e-12 else 0
        return float(m), float(v), n_ret, mu0, tau0_2, sigma2


def classify_source(bought_this_seller, rids, floor_oids: set[int],
                    bought_any_packet=False) -> str:
    """packet-backed iff this buyer bought this seller's packet; else trial-only.

    Platform-checked records are written separately (source=platform_checked).
    floor_oids / bought_any_packet are unused; kept so call sites stay unchanged.
    """
    if int(bought_this_seller) >= 1:
        return SRC_BACKED
    return SRC_TRIAL


def floor_oid_set(market: dict) -> dict[int, set[int]]:
    by_s = market.get("floor_offer_ids") or {}
    out: dict[int, set[int]] = {}
    for s, oids in by_s.items():
        out[int(s)] = {int(o) for o in oids}
    return out


@dataclass
class Record:
    seller: int
    pi_hat: float
    rids: tuple
    gammas: tuple
    source: str
    t: int
    buyer_id: int
    n_rows: int
    packets_bought: int = 0
    paid: float = 0.0


@dataclass
class QSourceStore:
    """Source-aware publisher. forced_trials uses skip_in_risk=False."""

    kind: str
    w_max: float = W_MAX
    delta: float = DELTA
    pkt_rows: int = PKT_ROWS
    prior: str = "eb"
    unw_when_no_backed: bool = False
    skip_in_risk: bool = False
    skip_threshold: float = 1.0
    records: list = field(default_factory=list)

    def add(self, rec: Record) -> None:
        self.records.append(rec)

    def _is_backed(self, rec) -> bool:
        return rec.source in BACKED_SOURCES

    def _backed_recs(self, sid: int) -> list:
        sid = int(sid)
        return [r for r in self.records if int(r.seller) == sid and self._is_backed(r)]

    def _backed_buyer_stats(self):
        n, s1, s2 = {}, {}, {}
        for rec in self.records:
            if not self._is_backed(rec):
                continue
            s = int(rec.seller)
            n[s] = n.get(s, 0) + 1
            s1[s] = s1.get(s, 0.0) + float(rec.pi_hat)
            s2[s] = s2.get(s, 0.0) + float(rec.pi_hat) ** 2
        return n, s1, s2

    def hyper_backed(self):
        """θ = (σ², μ0, τ0²) from packet-backed and platform-checked records."""
        n, s1, s2 = self._backed_buyer_stats()
        return hyper_from_backed(n, s1, s2, self.prior)

    def sellers(self) -> list[int]:
        return sorted({int(r.seller) for r in self.records})

    def recs_of(self, sid: int, src: str | None = None) -> list[Record]:
        sid = int(sid)
        out = []
        for r in self.records:
            if int(r.seller) != sid:
                continue
            if src is not None and r.source != src:
                continue
            out.append(r)
        return out

    def n_backed(self, sid: int) -> int:
        return int(len(self._backed_recs(sid)))

    def n_trial(self, sid: int) -> int:
        return int(len(self.recs_of(sid, SRC_TRIAL)))

    def rids_backed(self, sid: int) -> set:
        keys = set()
        for r in self._backed_recs(sid):
            keys.update(r.rids)
        return keys

    def rids_trial_all(self, sid: int) -> set:
        keys = set()
        for r in self.recs_of(sid, SRC_TRIAL):
            keys.update(r.rids)
        return keys

    def rids_overlap(self, sid: int) -> set:
        return self.rids_backed(sid) & self.rids_trial_all(sid)

    def n_overlap(self, sid: int) -> int:
        return int(len(self.rids_overlap(sid)))

    def rids_trial_exclusive(self, sid: int) -> set:
        return self.rids_trial_all(sid) - self.rids_backed(sid)

    def n_trial_rows(self, sid: int) -> int:
        if self.kind == KIND_ROW:
            return int(len(self.rids_trial_exclusive(sid)))
        return int(len(self.rids_trial_all(sid)))

    def n_backed_rows(self, sid: int) -> int:
        return int(len(self.rids_backed(sid)))

    def trial_weight(self, sid: int) -> float:
        return float(min(float(self.w_max), self.n_trial_rows(sid) / float(self.pkt_rows)))

    def _row_means(self, sid: int, src: str) -> dict:
        acc: dict[tuple, list[float]] = {}
        recs = self._backed_recs(sid) if src == SRC_BACKED else self.recs_of(sid, src)
        for r in recs:
            for rid, g in zip(r.rids, r.gammas):
                acc.setdefault(rid, []).append(float(g))
        return {k: float(np.mean(vs)) for k, vs in acc.items()}

    def _row_means_trial_exclusive(self, sid: int) -> dict:
        gset = set(self._row_means(sid, SRC_BACKED).keys())
        return {k: v for k, v in self._row_means(sid, SRC_TRIAL).items() if k not in gset}

    def pi_backed(self, sid: int) -> float:
        sigma2, mu0, tau0_2 = self.hyper_backed()
        if self.kind == KIND_ROW:
            means = self._row_means(sid, SRC_BACKED)
            return beta_mean([(g, 1.0) for g in means.values()])
        n, s1, _s2 = self._backed_buyer_stats()
        m, _v = publish_eb(n.get(int(sid), 0), s1.get(int(sid), 0.0), sigma2, mu0, tau0_2)
        return float(m)

    def pi_cap(self, sid: int) -> float:
        sigma2, mu0, tau0_2 = self.hyper_backed()
        wu = self.trial_weight(sid)
        if self.kind == KIND_ROW:
            gmeans = self._row_means(sid, SRC_BACKED)
            umeans = self._row_means_trial_exclusive(sid)
            e_u = max(len(umeans), 1)
            scale = min(1.0, (float(self.w_max) * float(self.pkt_rows)) / float(e_u)) if umeans else 0.0
            wts = [(g, 1.0) for g in gmeans.values()]
            wts.extend((g, scale) for g in umeans.values())
            return beta_mean(wts) if wts else float(mu0)
        n, s1, _s2 = self._backed_buyer_stats()
        u = [float(r.pi_hat) for r in self.recs_of(sid, SRC_TRIAL)]
        n_eff = float(n.get(int(sid), 0))
        s1_eff = float(s1.get(int(sid), 0.0))
        if u:
            n_eff += wu
            s1_eff += wu * float(np.mean(u))
        m, _v = publish_eb(n_eff, s1_eff, sigma2, mu0, tau0_2)
        return float(m)

    def pi_equal(self, sid: int) -> float:
        sigma2, mu0, tau0_2 = self.hyper_backed()
        if self.kind == KIND_ROW:
            gmeans = self._row_means(sid, SRC_BACKED)
            umeans = self._row_means_trial_exclusive(sid)
            wts = [(g, 1.0) for g in gmeans.values()]
            wts.extend((g, 1.0) for g in umeans.values())
            return beta_mean(wts) if wts else float(mu0)
        n, s1, _s2 = self._backed_buyer_stats()
        u = [float(r.pi_hat) for r in self.recs_of(sid, SRC_TRIAL)]
        n_eff = float(n.get(int(sid), 0) + len(u))
        s1_eff = float(s1.get(int(sid), 0.0) + (sum(u) if u else 0.0))
        m, _v = publish_eb(n_eff, s1_eff, sigma2, mu0, tau0_2)
        return float(m)

    def in_risk(self, sid: int) -> bool:
        if self.n_backed(sid) < 1 or self.n_trial(sid) < 1:
            return False
        return float(self.pi_backed(sid) - self.pi_equal(sid)) > float(self.delta)

    def published(self, sid: int) -> float:
        if self.in_risk(sid):
            return float(self.pi_backed(sid))
        if (
            self.unw_when_no_backed
            and self.kind == KIND_BUYER
            and self.n_backed(sid) == 0
            and self.n_trial(sid) >= 1
        ):
            return float(self.pi_equal(sid))
        return float(self.pi_cap(sid))

    def clip_bounds(self) -> tuple[float, float]:
        return CLIP_ROW if self.kind == KIND_ROW else CLIP_BUYER

    def published_clip(self, sid: int) -> float:
        lo, hi = self.clip_bounds()
        return float(np.clip(self.published(sid), lo, hi))

    def sigmahat(self) -> float:
        sigma2, _mu0, _tau = self.hyper_backed()
        return float(np.sqrt(max(float(sigma2), 1e-8)))

    def has_records(self, sid: int) -> bool:
        return (self.n_backed(sid) + self.n_trial(sid)) > 0

    def pi_map(self, vis_sids) -> dict[int, float]:
        return {
            int(s): self.published_clip(int(s))
            for s in vis_sids
            if self.has_records(int(s))
        }

    def skip_and_ntrials(self, vis_sids, B, beta, trial_cost):
        vis_sids = [int(s) for s in vis_sids]
        skip = set()
        for s in vis_sids:
            enough = (self.n_backed(s) + self.trial_weight(s)) >= float(self.skip_threshold) - 1e-12
            if enough and (self.skip_in_risk or not self.in_risk(s)):
                skip.add(int(s))
        n_vis = max(len(vis_sids), 1)
        n_insuf = sum(1 for s in vis_sids if s not in skip)
        if n_insuf == 0:
            n_trials = 0
        else:
            n_trials = max(
                1,
                int(np.floor(float(beta) * float(B) * n_insuf / n_vis / max(float(trial_cost), 1e-9))),
            )
        return int(n_trials), skip

    def backed_row_ids(self) -> dict[int, set]:
        out: dict[int, set] = {}
        for s in self.sellers():
            out[int(s)] = self.rids_backed(s)
        return out

    def snapshot(self, sid: int) -> dict:
        pg, pm, pu = self.pi_backed(sid), self.pi_cap(sid), self.pi_equal(sid)
        return dict(
            n_backed=self.n_backed(sid), n_trial=self.n_trial(sid),
            n_backed_rows=self.n_backed_rows(sid), n_trial_rows=self.n_trial_rows(sid),
            trial_weight=self.trial_weight(sid),
            n_overlap=self.n_overlap(sid),
            pi_backed=pg, pi_cap=pm, pi_equal=pu,
            pi_gap=float(pg - pu), in_risk=int(self.in_risk(sid)),
            published=self.published(sid),
        )
