"""incremental_exact matches full_recompute on a small stream."""

from __future__ import annotations

import numpy as np

from qsource.incremental import (
    FullRecompute,
    IncrementalExact,
    compare_exact_vs_log,
    make_stream,
    query_q1,
    query_q2,
    query_q3,
    rebuild_from_stream,
    synthetic_template,
)


def test_exact_matches_full_recompute():
    rng = np.random.default_rng([20261030, 2, 2, 200])
    tpl = synthetic_template(n_groups=40, recs_per=6, rng=rng)
    stream = make_stream(200, 80, rng, template=tpl)
    inc = IncrementalExact(200)
    log = FullRecompute(stream)
    for b in range(stream.n_buyers):
        inc.write_buyer(stream, b)
        log.write_buyer(stream, b)
    st = log.rebuild()
    cmp = compare_exact_vs_log(inc, st)
    assert cmp["ok"]
    assert cmp["max_numeric"] < 1e-12

    def _same(a, b):
        assert [x[0] for x in a] == [x[0] for x in b]
        if a:
            assert np.allclose([x[1] for x in a], [x[1] for x in b], atol=1e-12, rtol=0)

    _same(query_q1(inc, 0, 99), query_q1(st, 0, 99))
    _same(query_q2(inc), query_q2(st))
    _same(query_q3(inc), query_q3(st))
