#!/usr/bin/env python3
"""Sensitivity of QSource to Δ and w_max (reported in the main text, not Table III).

Paper setting: Amazon, buyer_level_floor, low_honest, instances 230–249,
(Δ, w_max) ∈ {(0.10, 0.25), (0.20, 0.25), (0.10, 1.0)} plus the main (0.05, 0.25).
"""

from __future__ import annotations

import argparse
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

from experiments.run_market import default_workers, parse_instances, write_result
from qsource.market import load_dataset
from qsource.simulate import run_instance

# Main setting is (0.05, 0.25). The other three are the paper sensitivity grid.
SETTINGS = (
    (0.05, 0.25),
    (0.10, 0.25),
    (0.20, 0.25),
    (0.10, 1.0),
)


def _job(args):
    dataset, design, config, inst, data_dir, w_max, delta = args
    bundle = load_dataset(dataset, data_dir)
    return run_instance(
        bundle, dataset, design, "qsource", config, "early", inst,
        w_max=w_max, delta=delta,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="amazon")
    ap.add_argument("--design", default="buyer_level_floor")
    ap.add_argument("--config", default="low_honest")
    ap.add_argument("--instances", default="230-249")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="results/sensitivity")
    ap.add_argument("--data", default=os.environ.get("QSOURCE_DATA", str(ROOT / "data")))
    args = ap.parse_args()
    insts = parse_instances(args.instances)
    jobs = []
    for ds in args.dataset.split(","):
        for design in args.design.split(","):
            for inst in insts:
                for delta, w_max in SETTINGS:
                    jobs.append((
                        ds.strip(), design.strip(), args.config, inst,
                        args.data, w_max, delta,
                    ))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    n_workers = int(args.workers) if args.workers is not None else default_workers()
    nw = max(1, min(n_workers, len(jobs)))
    rows = []
    with ProcessPoolExecutor(max_workers=nw) as ex:
        futs = {ex.submit(_job, job): job for job in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            write_result(out, r)
            m = r["meta"]
            hon = r["buyers"]
            hon = hon[hon["is_attacker"] == 0] if "is_attacker" in hon.columns else hon
            rows.append(dict(
                dataset=m["dataset"], design=m["design"], config=m["config"],
                instance=m["instance"], w_max=m["w_max"], delta=m["delta"],
                excess=float(hon["excess"].mean()) if len(hon) else float("nan"),
                packets=float(hon["packets"].sum()) if len(hon) else float("nan"),
                n_sold=int((hon.groupby("instance")["packets"].sum() > 0).sum()) if len(hon) else 0,
            ))
    pd.DataFrame(rows).to_csv(out / "sensitivity.csv", index=False)
    print(f"wrote {out / 'sensitivity.csv'} n={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
