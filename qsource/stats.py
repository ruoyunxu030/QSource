"""Paired bootstrap of the mean difference."""

from __future__ import annotations

import numpy as np

N_BOOT = 5000


def boot_mean(x, n_boot=N_BOOT, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, n=0)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return dict(
        mean=float(x.mean()), lo=float(np.quantile(m, 0.025)),
        hi=float(np.quantile(m, 0.975)), n=int(len(x)),
    )


def boot_diff(a, b, n_boot=N_BOOT, seed=0):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    if len(d) == 0:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, n=0, excludes0=False)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    lo, hi = float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))
    return dict(
        mean=float(d.mean()), lo=lo, hi=hi, n=int(len(d)),
        excludes0=bool(lo > 0 or hi < 0),
    )
