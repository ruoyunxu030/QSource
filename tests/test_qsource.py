"""QSource estimates and the two paper properties."""

from __future__ import annotations

import numpy as np

from qsource.attacks import insert_after
from qsource.const import (
    DELTA, KIND_BUYER, PKT_ROWS, RAISE_WEAK, SKIP_THRESHOLD,
    SRC_BACKED, SRC_PLATFORM, SRC_TRIAL,
)
from qsource.store import (
    ArrivalSkipStore,
    BuyerEvidenceStore,
    PurchaseWeightStore,
    QSourceStore,
    Record,
    RobustBuyerStore,
    classify_source,
    publish_eb,
)


def _store(w_max=0.25, delta=DELTA):
    return QSourceStore(kind=KIND_BUYER, w_max=w_max, delta=delta,
                        unw_when_no_backed=True, skip_in_risk=True)


def _rec(seller, pi, src, n_rows=40, t=1):
    rids = tuple((seller * 100 + t, i) for i in range(n_rows))
    return Record(
        seller=seller, pi_hat=float(pi), rids=rids,
        gammas=tuple([float(pi)] * n_rows), source=src,
        t=t, buyer_id=t, n_rows=n_rows,
    )


def test_three_estimates_eb():
    st = _store()
    st.add(_rec(0, 0.9, SRC_BACKED, t=1))
    st.add(_rec(0, 0.8, SRC_BACKED, t=2))
    st.add(_rec(0, 0.1, SRC_TRIAL, t=3))
    n, s1, s2 = st._backed_buyer_stats()
    sigma2, mu0, tau0_2 = st.hyper_backed()
    m_g, _ = publish_eb(n[0], s1[0], sigma2, mu0, tau0_2)
    assert abs(st.pi_backed(0) - m_g) < 1e-12
    wu = st.trial_weight(0)
    mean_u = 0.1
    m_mix, _ = publish_eb(n[0] + wu, s1[0] + wu * mean_u, sigma2, mu0, tau0_2)
    assert abs(st.pi_cap(0) - m_mix) < 1e-12
    m_unw, _ = publish_eb(n[0] + 1, s1[0] + 0.1, sigma2, mu0, tau0_2)
    assert abs(st.pi_equal(0) - m_unw) < 1e-12
    if st.in_risk(0):
        assert abs(st.published(0) - st.pi_backed(0)) < 1e-12
    else:
        assert abs(st.published(0) - st.pi_cap(0)) < 1e-12


def test_trial_weight_cap():
    """Prop. 1: trial-only shifts π_cap by at most w_max/(n_backed + w_max)."""
    w_max = 0.25
    st = _store(w_max=w_max)
    st.add(_rec(0, 0.95, SRC_BACKED, t=1))
    for t in range(2, 12):
        st.add(_rec(0, 0.0, SRC_TRIAL, t=t, n_rows=40))
    wu = st.trial_weight(0)
    assert wu <= w_max + 1e-12
    n_g = st.n_backed(0)
    assert n_g >= 1
    share = wu / (n_g + wu)
    assert share <= w_max / (1.0 + w_max) + 1e-12
    gap = abs(st.pi_cap(0) - st.pi_backed(0))
    bound = (w_max / (1.0 + w_max)) * abs(st.pi_backed(0) - 0.0)
    assert gap <= bound + 1e-9


def test_published_not_below_backed_minus_delta():
    delta = DELTA
    st = _store(w_max=1.0, delta=delta)
    st.add(_rec(0, 0.9, SRC_BACKED, t=1))
    st.add(_rec(0, 0.0, SRC_TRIAL, t=2))
    assert st.n_backed(0) >= 1
    assert st.w_max <= 1.0 + 1e-12
    assert st.published(0) >= st.pi_backed(0) - delta - 1e-12


def test_classify_source_packet_vs_trial():
    floor = {10, 11}
    assert classify_source(1, ((10, 0),), floor, bought_any_packet=0) == SRC_BACKED
    assert classify_source(0, ((10, 0),), floor, bought_any_packet=1) == SRC_TRIAL
    assert classify_source(0, ((10, 0),), floor, bought_any_packet=0) == SRC_TRIAL
    assert classify_source(0, ((99, 0),), floor, bought_any_packet=1) == SRC_TRIAL


def test_robust_median_and_trim():
    med = RobustBuyerStore(agg="median")
    trim = RobustBuyerStore(agg="trim")
    for t, v in enumerate((0.1, 0.5, 0.9), start=1):
        med.update_buyer(0, v, 40, t)
        trim.update_buyer(0, v, 40, t)
    m_med, *_ = med.publish(0)
    m_trim, _, n_trim, *_ = trim.publish(0)
    mean = RobustBuyerStore(agg="mean")
    for t, v in enumerate((0.1, 0.5, 0.9), start=1):
        mean.update_buyer(0, v, 40, t)
    # median of {0.1,0.5,0.9} is 0.5; trim of n=3 drops 0.1 and 0.9, n'=1
    assert n_trim == 1
    assert np.isfinite(m_med) and np.isfinite(m_trim)


def test_purchase_weight_formula():
    pw = PurchaseWeightStore()
    eq = BuyerEvidenceStore()
    # backed packet: w=1; trial-only 20 rows: w=0.5
    pw.update_buyer(0, 0.9, 40, 1, weight=1.0)
    pw.update_buyer(0, 0.1, 20, 2, weight=min(1.0, 20 / PKT_ROWS))
    eq.update_buyer(0, 0.9, 40, 1)
    eq.update_buyer(0, 0.1, 20, 2)
    # θ unweighted: same as Equal Weight
    assert pw._sigma2_tau() == eq._sigma2_tau()
    sigma2, mu0, tau0_2 = pw._sigma2_tau()
    n_w = 1.0 + 0.5
    loc = (1.0 * 0.9 + 0.5 * 0.1) / n_w
    m, v = publish_eb(n_w, n_w * loc, sigma2, mu0, tau0_2)
    got_m, got_v, n_ret, *_ = pw.publish(0)
    assert abs(got_m - m) < 1e-12
    assert abs(got_v - v) < 1e-12
    # n_ret is the π_override gate (means_from_buyer_store skips n<=0).
    # n_w=0.5→0, 1.5→2, 2.25→2.
    # Returning float n_w would change which sellers enter π and diverge
    # from the paper. skip_score remains the raw Σw.
    assert n_ret == 2  # n_w=1.5
    assert abs(pw.skip_score(0) - 1.5) < 1e-12
    # injection default is trial-only s/m
    inj = PurchaseWeightStore()
    inj.update_buyer(1, 0.5, 20, 0)
    assert abs(inj.skip_score(1) - 0.5) < 1e-12
    _m, _v, n0, *_ = inj.publish(1)
    assert n0 == 0  # n_w=0.5 < 1 → omitted from π_override
    pw225 = PurchaseWeightStore()
    pw225.update_buyer(2, 0.9, 40, 1, weight=1.0)
    pw225.update_buyer(2, 0.2, 20, 2, weight=0.5)
    pw225.update_buyer(2, 0.3, 30, 3, weight=0.75)
    _m, _v, n225, *_ = pw225.publish(2)
    assert abs(pw225.skip_score(2) - 2.25) < 1e-12
    assert n225 == 2  # n_w=2.25 → round 2, still ≥1 so published


def test_skip_threshold_default_matches_one():
    assert abs(float(SKIP_THRESHOLD) - 1.0) < 1e-12
    st = _store()
    st.add(_rec(0, 0.9, SRC_BACKED, t=1))
    n_trials, skip = st.skip_and_ntrials([0, 1], B=20, beta=0.35, trial_cost=0.5)
    st.skip_threshold = 1.0
    n2, skip2 = st.skip_and_ntrials([0, 1], B=20, beta=0.35, trial_cost=0.5)
    assert n_trials == n2 and skip == skip2
    assert 0 in skip
    st.skip_threshold = 2.0
    _n3, skip3 = st.skip_and_ntrials([0, 1], B=20, beta=0.35, trial_cost=0.5)
    assert 0 not in skip3


def test_raise_weak_const():
    assert abs(RAISE_WEAK["amazon"] - 0.961628827221886) < 1e-12
    assert abs(RAISE_WEAK["domainnet"] - 0.9808480688625696) < 1e-12


def test_platform_checked_counts_as_backed():
    st = _store()
    st.add(_rec(0, 0.8, SRC_PLATFORM, t=1))
    assert st.n_backed(0) == 1
    assert st.n_trial(0) == 0
    st.add(_rec(0, 0.1, SRC_TRIAL, t=2))
    assert st.n_backed(0) == 1
    n_trials, skip = st.skip_and_ntrials([0], B=20, beta=0.35, trial_cost=0.5)
    assert 0 in skip


def test_insert_before_first():
    counts = insert_after(0.1, "before_first", n_honest=30)
    assert counts == {0: 3}


def test_skip_threshold_two_counts_write_times():
    st = ArrivalSkipStore(prior="eb")
    st.update_buyer(0, 0.9, 40, t=1)
    st.update_buyer(0, 0.8, 40, t=1)
    assert st.n_buyers(0) == 2
    assert abs(st.skip_score(0) - 1.0) < 1e-12
    st.update_buyer(0, 0.85, 40, t=2)
    assert abs(st.skip_score(0) - 2.0) < 1e-12
