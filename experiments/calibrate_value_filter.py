#!/usr/bin/env python3
"""Leave-one-instance 1% quantile of honest records from config=none runs.

For each (dataset, design, instance), drop that instance and take the 1%
quantile of the remaining honest record values. Value Filter reads this table;
a missing row is an error, not a disabled filter.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _pi_col(df: pd.DataFrame) -> str:
    if "pi_hat" in df.columns:
        return "pi_hat"
    if "pi_hat_jb" in df.columns:
        return "pi_hat_jb"
    raise ValueError("records need pi_hat or pi_hat_jb")


def _design_col(df: pd.DataFrame) -> pd.Series:
    if "design" in df.columns:
        return df["design"].astype(str)
    raise ValueError("records need a design column")


def load_none_records(results: Path) -> pd.DataFrame:
    parts = []
    cells = results / "cells"
    cell_files = []
    if cells.is_dir():
        cell_files = [p for p in sorted(cells.glob("*/records.csv")) if p.stat().st_size > 0]
    if cell_files:
        parts = [pd.read_csv(p) for p in cell_files]
    else:
        top = results / "records.csv"
        if top.exists() and top.stat().st_size > 0:
            parts.append(pd.read_csv(top))
    if not parts:
        raise SystemExit(f"no records.csv under {results}")
    df = pd.concat(parts, ignore_index=True)
    if "config" in df.columns:
        df = df[df["config"].astype(str) == "none"]
    if "method" in df.columns:
        df = df[df["method"].astype(str) == "equal_weight"]
    if "is_attacker" in df.columns:
        df = df[df["is_attacker"].fillna(0).astype(int) == 0]
    elif "is_writer" in df.columns:
        df = df[df["is_writer"].fillna(0).astype(int) == 0]
    return df


def loo_thresholds(records: pd.DataFrame, instances=None) -> pd.DataFrame:
    """Honest-record 1% quantile, leave-one-instance, per dataset and design.

    Value Filter is calibrated on Equal Weight only. Rows with a method column
    other than equal_weight are dropped.
    """
    df = records.copy()
    if "method" in df.columns:
        df = df[df["method"].astype(str) == "equal_weight"]
    if "is_attacker" in df.columns:
        df = df[df["is_attacker"].fillna(0).astype(int) == 0]
    elif "is_writer" in df.columns:
        df = df[df["is_writer"].fillna(0).astype(int) == 0]
    df["design"] = _design_col(df)
    pi = _pi_col(df)
    df["_pi"] = pd.to_numeric(df[pi], errors="coerce")
    df = df[np.isfinite(df["_pi"])]
    if instances is None:
        instances = sorted(int(i) for i in df["instance"].unique())
    else:
        instances = [int(i) for i in instances]
    rows = []
    for ds, gds in df.groupby(df["dataset"].astype(str), sort=False):
        for design, g in gds.groupby(gds["design"].astype(str), sort=False):
            for inst in instances:
                sl = g[g["instance"].astype(int) != int(inst)]
                x = sl["_pi"].to_numpy(float)
                x = x[np.isfinite(x)]
                cal = sorted(int(i) for i in sl["instance"].unique()) if len(sl) else []
                rows.append(dict(
                    dataset=str(ds), design=str(design), instance=int(inst),
                    threshold=float(np.quantile(x, 0.01)) if len(x) else np.nan,
                    n_records=int(len(x)),
                    n_cal_instances=int(len(cal)),
                ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results",
                    help="directory with config=none records.csv or cells/*/records.csv")
    ap.add_argument("--out", default="")
    ap.add_argument("--instances", default="",
                    help="instances to write thresholds for; default = those in the records")
    args = ap.parse_args()
    root = Path(args.results)
    rec = load_none_records(root)
    insts = None
    if args.instances:
        from experiments.run_market import parse_instances
        insts = parse_instances(args.instances)
    tab = loo_thresholds(rec, insts)
    out = Path(args.out) if args.out else root / "thresholds.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out, index=False)
    print(f"wrote {out} n={len(tab)}", flush=True)


if __name__ == "__main__":
    main()
