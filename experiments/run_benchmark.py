#!/usr/bin/env python3
"""Storage benchmark matching the paper Table IV / Figure 5 load.

Template: config=none forced_trials records of buyer_level and buyer_level_floor
(the same source as the paper). Default N = 1000,10000,100000; 5 query repeats
on one write stream; Q1 100 random segments; Q2/Q3 20 each. Timing numbers
depend on the machine. Use --synthetic only for a smoke test without records.

Writes:
  latency.csv          query latency (s) per store / query / repeat
  write_latency.csv    per-buyer write latency (s)
  state_size.csv       persistent state from nbytes_parts (MB)
  snapshot_error.csv   snapshot vs exact at periodic buyers
  stream_stats.csv     stream counts plus end-of-stream exact-vs-log
  correctness.json     checkpoint exactness (seller stats, θ, published,
                       lower bound, risk, and ranking/threshold/risk queries)
  table_iv_raw.csv     paper Table IV columns
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qsource.incremental import (
    IncrementalExact,
    IncrementalSnapshot,
    FullRecompute,
    REFRESH_EVERY,
    SEED,
    SEG,
    checkpoint_indices,
    compare_exact_vs_log,
    make_stream,
    n_buyers_for,
    rebuild_from_stream,
    synthetic_template,
    template_from_records,
)

STORE_NAME = {
    "full_recompute": FullRecompute,
    "incremental_exact": IncrementalExact,
    "incremental_snapshot": IncrementalSnapshot,
}
STORE_LABEL = {
    "full_recompute": "Full recompute",
    "incremental_exact": "Incremental exact",
    "incremental_snapshot": "Incremental snapshot",
}
QUERY_NAME = {"q1": "ranking", "q2": "threshold", "q3": "risk_review"}

PAPER_DESIGNS = ("buyer_level", "buyer_level_floor")
N_Q1 = 100
N_Q2 = 20
N_Q3 = 20
N_REP = 5
ATOL = 1e-12


def _collect_record_files(records_dir: Path) -> list[Path]:
    """Prefer cells/*/records.csv. Use a top-level records.csv only if no cells."""
    if records_dir.is_file():
        return [records_dir] if records_dir.stat().st_size > 0 else []
    cells = sorted(
        p for p in records_dir.glob("cells/*/records.csv") if p.stat().st_size > 0
    )
    if cells:
        return cells
    files = []
    top = records_dir / "records.csv"
    if top.exists() and top.stat().st_size > 0:
        files.append(top)
    for p in sorted(records_dir.glob("*records*.csv")):
        if p.resolve() == top.resolve():
            continue
        if p.stat().st_size > 0:
            files.append(p)
    return files


def load_paper_template(records_dir: Path) -> dict:
    files = _collect_record_files(records_dir)
    if not files:
        raise SystemExit(
            f"no records under {records_dir}; run config=none forced_trials "
            "on buyer_level and buyer_level_floor, or pass --synthetic"
        )
    kept = []
    for p in files:
        df = pd.read_csv(p, nrows=1)
        cols = set(df.columns)
        ok = True
        if "config" in cols:
            full = pd.read_csv(p, usecols=lambda c: c in ("config", "method", "design", "policy"))
            if "config" in full.columns:
                full = full[full["config"].astype(str) == "none"]
            if "method" in full.columns:
                full = full[full["method"].astype(str) == "forced_trials"]
            if "design" in full.columns:
                full = full[full["design"].astype(str).isin(PAPER_DESIGNS)]
            if not len(full):
                ok = False
        if ok:
            kept.append(p)
    if not kept:
        raise SystemExit(
            f"no config=none forced_trials records of {PAPER_DESIGNS} under {records_dir}"
        )
    return template_from_records(
        kept, config="none", method="forced_trials",
        designs=PAPER_DESIGNS,
    )


def _time_queries(store, n_sellers, rng, is_log=False):
    n_seg = max(n_sellers // SEG, 1)
    segs = [int(x) for x in rng.integers(0, n_seg, size=N_Q1)]
    rows = []
    for si in segs:
        lo = si * SEG
        hi = min(lo + SEG - 1, n_sellers - 1)
        dt = float(store.q1(lo, hi)[0])
        rows.append(("ranking", dt))
    for _ in range(N_Q2):
        rows.append(("threshold", float(store.q2()[0])))
    for _ in range(N_Q3):
        rows.append(("risk_review", float(store.q3()[0])))
    return rows


def _state_mb(name: str, parts: dict) -> float:
    """Persistent state. Full recompute is the log only, not the query rebuild."""
    if name == "full_recompute":
        return float(parts.get("log_mb") or 0.0)
    return float(parts.get("total_mb") or 0.0)


def _ckpt_summary(cmp: dict) -> dict:
    q = cmp.get("queries") or {}
    q_ok = True
    for name in ("q1", "q2", "q3"):
        rec = q.get(name) or {}
        if int(rec.get("set_mismatch") or 0) or int(rec.get("order_mismatch") or 0):
            q_ok = False
        if float(rec.get("val_err") or 0.0) > ATOL:
            q_ok = False
    return dict(
        ok=bool(cmp.get("ok")) and q_ok and float(cmp.get("max_numeric") or 0.0) <= ATOL,
        max_numeric=float(cmp.get("max_numeric") or 0.0),
        int_mismatch=int(cmp.get("int_mismatch") or 0),
        query_mismatch=int(cmp.get("query_mismatch") or 0),
        queries_consistent=bool(q_ok),
        q1_set=int((q.get("q1") or {}).get("set_mismatch") or 0),
        q1_order=int((q.get("q1") or {}).get("order_mismatch") or 0),
        q1_val=float((q.get("q1") or {}).get("val_err") or 0.0),
        q2_set=int((q.get("q2") or {}).get("set_mismatch") or 0),
        q2_order=int((q.get("q2") or {}).get("order_mismatch") or 0),
        q2_val=float((q.get("q2") or {}).get("val_err") or 0.0),
        q3_set=int((q.get("q3") or {}).get("set_mismatch") or 0),
        q3_order=int((q.get("q3") or {}).get("order_mismatch") or 0),
        q3_val=float((q.get("q3") or {}).get("val_err") or 0.0),
    )


def run_one(n_sellers, tpl, rng, n_rep=N_REP):
    n_buyers = n_buyers_for(n_sellers, float(tpl["recs_per_buyer"]) or 8.0)
    stream = make_stream(n_sellers, n_buyers, rng, template=tpl)
    n_ckpt = 3 if int(n_sellers) >= 100_000 else 20
    ckpt_set = set(checkpoint_indices(n_buyers, n_ckpt))
    err_rows = []
    write_rows = []
    ckpt_rows = []
    inc = IncrementalExact(n_sellers)
    snap = IncrementalSnapshot(n_sellers)
    log = FullRecompute(stream)
    for b in range(n_buyers):
        dt_inc = float(inc.write_buyer(stream, b))
        dt_log = float(log.write_buyer(stream, b))
        dt_snap = float(snap.write_buyer(stream, b))
        write_rows.append(dict(n_sellers=int(n_sellers), store="incremental_exact",
                               buyer=int(b), write_s=dt_inc))
        write_rows.append(dict(n_sellers=int(n_sellers), store="full_recompute",
                               buyer=int(b), write_s=dt_log))
        write_rows.append(dict(n_sellers=int(n_sellers), store="incremental_snapshot",
                               buyer=int(b), write_s=dt_snap))
        refreshed = snap.maybe_refresh()
        if b in ckpt_set:
            rebuilt = rebuild_from_stream(stream, inc.n_buyers)
            cmp = compare_exact_vs_log(inc, rebuilt)
            rec = _ckpt_summary(cmp)
            rec.update(n_sellers=int(n_sellers), buyer=int(b), n_buyers_so_far=int(inc.n_buyers))
            ckpt_rows.append(rec)
        if n_sellers >= 1000 and (b + 1) % max(n_buyers // 20, 1) == 0:
            from qsource.incremental import publish_arrays
            th = inc.theta.hyper()
            parts = publish_arrays(inc.n_g, inc.s1_g, inc.n_u, inc.s1_u, inc.e_u, *th)
            mae = float(np.nanmean(np.abs(snap.mat_pub - parts["pub"]))) if snap._fresh else np.nan
            err_rows.append(dict(
                n_sellers=int(n_sellers), buyer=int(b + 1),
                refresh=int(refreshed > 0 or snap.n_buyers % REFRESH_EVERY == 0),
                mean_abs_error=mae,
            ))
    st = log.rebuild()
    cmp_end = compare_exact_vs_log(inc, st)
    end_sum = _ckpt_summary(cmp_end)
    state_rows = []
    for name, obj in (
        ("full_recompute", log),
        ("incremental_exact", inc),
        ("incremental_snapshot", snap),
    ):
        parts = obj.nbytes_parts()
        state_rows.append(dict(
            n_sellers=int(n_sellers), store=name, state_mb=_state_mb(name, parts),
            **{k: parts.get(k) for k in (
                "seller_mb", "bitmap_mb", "acc_mb", "cand_mb", "log_mb", "snapshot_mb", "total_mb",
            )},
        ))
    lat_rows = []
    for rep in range(int(n_rep)):
        qrng = np.random.default_rng(SEED + [int(n_sellers), int(rep), 7])
        for store_name, obj in (
            ("incremental_exact", inc),
            ("incremental_snapshot", snap),
            ("full_recompute", log),
        ):
            for query, dt in _time_queries(obj, n_sellers, qrng, is_log=(store_name == "full_recompute")):
                lat_rows.append(dict(
                    n_sellers=int(n_sellers), store=store_name, query=query,
                    latency_s=float(dt), rep=int(rep),
                ))
    ckpt_ok = bool(ckpt_rows) and all(r["ok"] for r in ckpt_rows)
    max_ckpt = float(max((r["max_numeric"] for r in ckpt_rows), default=0.0))
    stream_row = dict(
        n_sellers=int(stream.n_sellers),
        n_buyers=int(stream.n_buyers),
        n_recs=int(stream.n_recs),
        n_g=int(stream.n_g),
        n_u=int(stream.n_u),
        g_frac=float(stream.g_frac),
        recs_per_buyer=float(stream.recs_per_buyer),
        tpl_n_files=int(tpl.get("n_files") or 0),
        tpl_n_groups=int(tpl.get("n_groups") or 0),
        tpl_recs_per_buyer=float(tpl.get("recs_per_buyer") or 0),
        tpl_g_frac=float(tpl.get("g_frac") or 0),
        exact_vs_full_ok=bool(end_sum["ok"]),
        max_numeric=float(end_sum["max_numeric"]),
        n_checkpoints=int(len(ckpt_rows)),
        checkpoints_ok=bool(ckpt_ok),
        checkpoint_max_numeric=max_ckpt,
        queries_consistent=bool(end_sum["queries_consistent"] and all(r["queries_consistent"] for r in ckpt_rows)),
    )
    correctness = dict(
        n_sellers=int(n_sellers),
        n_buyers=int(n_buyers),
        n_ckpt_requested=int(n_ckpt),
        n_checkpoints=int(len(ckpt_rows)),
        n_ok=int(sum(1 for r in ckpt_rows if r["ok"])),
        max_numeric=max_ckpt,
        queries_consistent=bool(stream_row["queries_consistent"]),
        end_of_stream=end_sum,
        checkpoints=ckpt_rows,
    )
    return lat_rows, err_rows, stream_row, write_rows, state_rows, correctness


def _table_iv_rows(lat, writes, state) -> list[dict]:
    rows = []
    lat_df = pd.DataFrame(lat)
    wr_df = pd.DataFrame(writes)
    st_df = pd.DataFrame(state)
    if lat_df.empty:
        return rows
    for n in sorted(int(x) for x in lat_df.n_sellers.unique()):
        for store in ("full_recompute", "incremental_exact", "incremental_snapshot"):
            sl = lat_df[(lat_df.n_sellers == n) & (lat_df.store == store)]
            if sl.empty:
                continue
            def _med_ms(q):
                x = sl[sl["query"] == q]["latency_s"]
                return float(np.median(x) * 1000.0) if len(x) else np.nan
            w = wr_df[(wr_df.n_sellers == n) & (wr_df.store == store)]["write_s"] if len(wr_df) else pd.Series(dtype=float)
            st = st_df[(st_df.n_sellers == n) & (st_df.store == store)]
            rows.append(dict(
                N=int(n),
                Store=STORE_LABEL[store],
                Write_us=float(np.median(w) * 1e6) if len(w) else np.nan,
                Rank_ms=_med_ms("ranking"),
                Thresh_ms=_med_ms("threshold"),
                Risk_ms=_med_ms("risk_review"),
                State_MB=float(st["state_mb"].iloc[0]) if len(st) else np.nan,
            ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="1000,10000,100000")
    ap.add_argument("--out", default="results/benchmark")
    ap.add_argument("--records", default="",
                    help="directory of run_market output (config=none forced_trials records)")
    ap.add_argument("--synthetic", action="store_true",
                    help="use a synthetic template (not the paper load)")
    ap.add_argument("--reps", type=int, default=N_REP)
    args = ap.parse_args()
    ns = tuple(int(x) for x in args.n.split(",") if x.strip())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        tpl = synthetic_template()
    else:
        if not args.records:
            raise SystemExit(
                "pass --records DIR of run_market config=none forced_trials output, "
                "or --synthetic for a smoke-test template"
            )
        tpl = load_paper_template(Path(args.records))
    print(
        f"template n_groups={tpl['n_groups']} recs_per_buyer={tpl['recs_per_buyer']:.6f} "
        f"g_frac={tpl['g_frac']:.6f} n_files={tpl['n_files']}",
        flush=True,
    )
    lat, err, streams, writes, state, correctness = [], [], [], [], [], []
    for n in ns:
        rng = np.random.default_rng(SEED + [int(n)])
        print(f"benchmark N={n}", flush=True)
        a, b, s, w, st, c = run_one(int(n), tpl, rng, n_rep=int(args.reps))
        lat.extend(a)
        err.extend(b)
        streams.append(s)
        writes.extend(w)
        state.extend(st)
        correctness.append(c)
        print(
            f"  buyers={s['n_buyers']} recs={s['n_recs']} backed={s['n_g']} "
            f"trial={s['n_u']} exact_ok={s['exact_vs_full_ok']} "
            f"ckpt={s['n_checkpoints']} ckpt_ok={s['checkpoints_ok']} "
            f"max_num={s['checkpoint_max_numeric']:.3e}",
            flush=True,
        )
        for row in st:
            print(f"  state {row['store']} {row['state_mb']:.4f} MB", flush=True)
    pd.DataFrame(lat).to_csv(out / "latency.csv", index=False)
    pd.DataFrame(err).to_csv(out / "snapshot_error.csv", index=False)
    pd.DataFrame(streams).to_csv(out / "stream_stats.csv", index=False)
    pd.DataFrame(writes).to_csv(out / "write_latency.csv", index=False)
    pd.DataFrame(state).to_csv(out / "state_size.csv", index=False)
    t4 = _table_iv_rows(lat, writes, state)
    pd.DataFrame(t4).to_csv(out / "table_iv_raw.csv", index=False)
    (out / "correctness.json").write_text(
        json.dumps(dict(cells=streams, checkpoints=correctness, template=dict(
            n_files=int(tpl.get("n_files") or 0),
            n_groups=int(tpl.get("n_groups") or 0),
            n_template_recs=int(tpl.get("n_template_recs") or 0),
            recs_per_buyer=float(tpl.get("recs_per_buyer") or 0),
            g_frac=float(tpl.get("g_frac") or 0),
        )), indent=2) + "\n"
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
