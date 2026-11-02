#!/usr/bin/env python3
"""Run market instances: --dataset --design --method --config --timing --instances --workers --out.

Each cell (dataset, design, method, config, timing, instance) is written to its own
directory under out/cells/ and also appended to the shared CSVs, de-duplicated by
those keys. Re-running a cell replaces that cell only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _k in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_k, "1")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qsource.const import DELTA, DESIGNS, SKIP_THRESHOLD, W_MAX
from qsource.market import load_dataset
from qsource.simulate import run_instance

METHODS = (
    "equal_weight", "value_filter", "forced_trials", "forced_repeat_trials",
    "qsource", "median", "trimmed_mean", "purchase_weight",
)
METHOD_ALIAS = {"forced_repeat_trials": "forced_trials"}
DESIGN_ALIAS = {"known_reliability": "known_quality"}
CELL_KEYS = ("dataset", "design", "method", "config", "timing", "instance")
CELL_EXTRA = ("skip_threshold", "target_goal", "w_max", "delta")
TABLE_KEYS = {
    "buyers": CELL_KEYS + CELL_EXTRA + ("buyer", "is_attacker"),
    "pays": CELL_KEYS + CELL_EXTRA + ("buyer", "seller", "is_attacker"),
    "posts": CELL_KEYS + CELL_EXTRA + ("buyer", "seller", "is_attacker"),
    "records": CELL_KEYS + CELL_EXTRA + ("buyer", "seller", "is_attacker"),
    "traj": CELL_KEYS + CELL_EXTRA + ("buyer",),
    "plat": CELL_KEYS + CELL_EXTRA + ("buyer", "seller"),
}


def parse_instances(s: str) -> list[int]:
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def available_cpus() -> int:
    n = None
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = None
    for path in ("/sys/fs/cgroup/cpu.max",):
        try:
            quota, period = Path(path).read_text().split()
            if quota != "max" and int(period) > 0:
                n_cg = max(1, int(int(quota) / int(period)))
                n = n_cg if n is None else min(n, n_cg)
        except (FileNotFoundError, ValueError, OSError):
            pass
    return int(n) if n else max(1, os.cpu_count() or 2)


def default_workers(reserve: int = 4) -> int:
    return max(1, available_cpus() - int(reserve))


def cell_stem(meta: dict) -> str:
    base = "_".join(str(meta[k]) for k in CELL_KEYS)
    k = float(meta.get("skip_threshold", SKIP_THRESHOLD) or SKIP_THRESHOLD)
    if abs(k - float(SKIP_THRESHOLD)) > 1e-12:
        base += f"_skip{k:g}"
    if str(meta.get("config")) == "none" and str(meta.get("target_goal") or "lower") == "raise":
        base += "_weak"
    w = float(meta.get("w_max", W_MAX) if meta.get("w_max") is not None else W_MAX)
    d = float(meta.get("delta", DELTA) if meta.get("delta") is not None else DELTA)
    if abs(w - float(W_MAX)) > 1e-12:
        base += f"_w{w:g}"
    if abs(d - float(DELTA)) > 1e-12:
        base += f"_d{d:g}"
    return base


def load_thresholds(path: Path) -> dict:
    df = pd.read_csv(path)
    need = {"dataset", "design", "instance", "threshold"}
    if not need.issubset(df.columns):
        raise SystemExit(f"{path} must have columns {sorted(need)}")
    out = {}
    for r in df.itertuples(index=False):
        out[(str(r.dataset), str(r.design), int(r.instance))] = float(r.threshold)
    return out


def _cat(dfs):
    dfs = [d for d in dfs if d is not None and len(d)]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _upsert_csv(path: Path, new: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame:
    if new is None or not len(new):
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()
    have = [c for c in keys if c in new.columns]
    if path.exists() and path.stat().st_size > 0:
        old = pd.read_csv(path)
        both = pd.concat([old, new], ignore_index=True)
    else:
        both = new
    if have and len(both):
        both = both.drop_duplicates(subset=have, keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    both.to_csv(path, index=False)
    return both


def _upsert_meta(path: Path, new_rows: list[dict]) -> list[dict]:
    old = []
    if path.exists() and path.stat().st_size > 0:
        try:
            blob = json.loads(path.read_text())
            old = blob if isinstance(blob, list) else [blob]
        except json.JSONDecodeError:
            old = []
    by = {}
    for row in old + list(new_rows):
        key = tuple(str(row.get(k, "")) for k in CELL_KEYS)
        by[key] = row
    rows = list(by.values())
    path.write_text(json.dumps(rows, indent=2, default=str) + "\n")
    return rows


def write_result(out: Path, result: dict) -> None:
    meta = result["meta"]
    cell = out / "cells" / cell_stem(meta)
    cell.mkdir(parents=True, exist_ok=True)
    tables = {
        "buyers": result["buyers"],
        "pays": result["pays"],
        "posts": result["posts"],
        "records": result["records"],
        "traj": result["traj"],
        "plat": result.get("plat"),
    }
    names = {
        "buyers": "buyers.csv",
        "pays": "pays.csv",
        "posts": "posts.csv",
        "records": "records.csv",
        "traj": "trajectories.csv",
        "plat": "plat.csv",
    }
    for key, df in tables.items():
        p = cell / names[key]
        if df is not None and len(df):
            df.to_csv(p, index=False)
        elif not p.exists():
            pd.DataFrame().to_csv(p, index=False)
        _upsert_csv(out / names[key], df, TABLE_KEYS[key])
    (cell / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    _upsert_meta(out / "meta.json", [meta])


def _job(args):
    (dataset, design, method, config, timing, inst, data_dir, q01,
     w_max, delta, skip_threshold, target_goal) = args
    bundle = load_dataset(dataset, data_dir)
    return run_instance(
        bundle, dataset, design, method, config, timing, inst,
        q01=q01, w_max=w_max, delta=delta,
        skip_threshold=skip_threshold, target_goal=target_goal,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["amazon", "domainnet"])
    ap.add_argument("--design", required=True, choices=list(DESIGNS) + list(DESIGN_ALIAS))
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--config", default="none",
                    choices=["none", "low_honest", "mediocre", "fixed", "extreme",
                             "extreme_one_account", "raise_weak"])
    ap.add_argument(
        "--timing", default="early",
        choices=["early", "late", "before_first"],
        help="early: injected accounts in buyers 1–10; late: buyers 21–30; "
             "before_first: all accounts write before buyer 1 (Fig. 1).",
    )
    ap.add_argument("--skip-threshold", type=float, default=SKIP_THRESHOLD,
                    help="The amount of evidence a seller needs before a buyer may skip "
                         "its trial (default 1). Equal Weight counts records, Purchase "
                         "Weight sums record weights, and QSource uses n^backed + w^trial.")
    ap.add_argument("--target-goal", default="", choices=["", "lower", "raise"],
                    help="Target seller: π=1 (lower) or first π=0.5 (raise). "
                         "Default lower, except raise_weak uses raise. "
                         "none + raise is the raise_weak control.")
    ap.add_argument("--instances", default="230-231")
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel instances. Default: available CPUs minus 4.")
    ap.add_argument("--out", default="results")
    ap.add_argument("--data", default=os.environ.get("QSOURCE_DATA", str(ROOT / "data")))
    ap.add_argument("--thresholds", default="",
                    help="CSV with dataset,design,instance,threshold (required for value_filter)")
    ap.add_argument("--w-max", type=float, default=None,
                    help="Trial-only weight cap (paper default 0.25).")
    ap.add_argument("--delta", type=float, default=None,
                    help="Risk-set gap (paper default 0.05).")
    args = ap.parse_args()
    args.method = METHOD_ALIAS.get(args.method, args.method)
    args.design = DESIGN_ALIAS.get(args.design, args.design)
    insts = parse_instances(args.instances)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cells").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    thr = {}
    if args.thresholds:
        thr = load_thresholds(Path(args.thresholds))
    if args.method == "value_filter":
        missing = [
            inst for inst in insts
            if (args.dataset, args.design, int(inst)) not in thr
        ]
        if missing:
            raise SystemExit(
                "value_filter reads one threshold per instance from --thresholds; "
                f"missing {args.dataset}/{args.design} instances {missing}. "
                "Run experiments/calibrate_value_filter.py on equal_weight config=none runs."
            )
    jobs = []
    for inst in insts:
        q01 = None
        if args.method == "value_filter":
            q01 = thr[(args.dataset, args.design, int(inst))]
        jobs.append((
            args.dataset, args.design, args.method, args.config, args.timing, inst,
            args.data, q01,
            args.w_max if args.w_max is not None else W_MAX,
            args.delta if args.delta is not None else DELTA,
            float(args.skip_threshold),
            args.target_goal or None,
        ))
    n_workers = int(args.workers) if args.workers is not None else default_workers()
    nw = max(1, min(n_workers, len(jobs)))
    results = []
    if nw == 1:
        for job in jobs:
            results.append(_job(job))
    else:
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futs = [ex.submit(_job, job) for job in jobs]
            for fut in as_completed(futs):
                results.append(fut.result())
    for r in results:
        write_result(out, r)
    print(f"wrote {out} n={len(results)} workers={nw}", flush=True)


if __name__ == "__main__":
    main()
