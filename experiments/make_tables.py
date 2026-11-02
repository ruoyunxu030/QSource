#!/usr/bin/env python3
"""Build Table I–IV, floor cost, and additional-experiment summaries.

Paper tables (ICDE 2027):
  Table I   equal-weight sales by sharing design and injection config
  Table II  methods under buyer_level and buyer_level_floor (plus median /
            trimmed_mean on buyer_level_floor)
  Table III excess-loss change vs equal weight when config=none
  Table IV  store benchmark stream size (latency is machine-dependent)
Additional experiments: raise_weak vs same-target none, skip-threshold 2 vs 1,
Purchase Weight vs QSource excess on buyer_level_floor.
Sales are the sum of target packets over 30 honest buyers, then the mean over instances.
Paired deltas use 5000 bootstrap, 95% interval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qsource.stats import boot_diff, boot_mean

CELL = ["dataset", "design", "config", "method", "instance"]
MAIN_DELTA, MAIN_WMAX = 0.05, 0.25
TABLE_I_CONFIGS = [
    "none", "low_honest", "mediocre", "fixed", "extreme", "extreme_late",
    "extreme_one_account",
]
TABLE_I_COL = {
    "none": "None",
    "low_honest": "Low Honest",
    "mediocre": "Mediocre",
    "fixed": "Fixed",
    "extreme": "Extreme",
    "extreme_late": "Extreme late",
    "extreme_one_account": "Extreme (1 acct)",
}
STORE_LABEL = {
    "full_recompute": "Full recompute",
    "incremental_exact": "Incremental exact",
    "incremental_snapshot": "Incremental snapshot",
}


def _read_all(root: Path, name: str) -> pd.DataFrame:
    parts = []
    top = root / name
    if top.exists() and top.stat().st_size > 0:
        parts.append(pd.read_csv(top))
    cells = root / "cells"
    if cells.is_dir():
        for p in sorted(cells.glob(f"*/{name}")):
            if p.stat().st_size > 0:
                parts.append(pd.read_csv(p))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    keys = [c for c in (
        "dataset", "design", "method", "config", "timing", "instance",
        "skip_threshold", "target_goal",
        "buyer", "seller", "is_attacker",
    ) if c in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys, keep="last")
    return df


def _honest(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "is_attacker" in df.columns:
        df = df[df["is_attacker"].fillna(0).astype(int) == 0]
    if "buyer" in df.columns:
        df = df[df["buyer"].between(1, 30)]
    return df


def _instance_sales(traj: pd.DataFrame) -> pd.DataFrame:
    hon = _honest(traj)
    if hon.empty or "packets" not in hon.columns:
        return pd.DataFrame()
    keys = [c for c in CELL if c in hon.columns]
    for extra in ("timing", "skip_threshold", "target_goal"):
        if extra in hon.columns and extra not in keys:
            keys.append(extra)
    return hon.groupby(keys, sort=False)["packets"].sum().reset_index(name="sales")


def _cell(sales: pd.Series) -> dict:
    b = boot_mean(sales.values)
    n_adm = int((sales.values > 0).sum())
    return dict(
        packets_mean=b["mean"], packets_lo=b["lo"], packets_hi=b["hi"],
        n=b["n"], n_admitted=n_adm,
    )


def _fmt_sales(mean, n_adm, n):
    if not np.isfinite(mean):
        return "--"
    if n_adm == n:
        return f"{mean:.2f}"
    return f"{mean:.2f} ({n_adm}/{n})"


def _timing(df: pd.DataFrame) -> pd.Series:
    if "timing" in df.columns:
        t = df["timing"].fillna("early").astype(str)
        t = t.where(t != "", "early")
        return t
    return pd.Series("early", index=df.index)


def table_i(traj: pd.DataFrame) -> pd.DataFrame:
    sales = _instance_sales(traj)
    if sales.empty:
        return sales
    sales = sales[sales["method"] == "equal_weight"] if "method" in sales.columns else sales
    sales = sales.copy()
    sales["timing"] = _timing(sales)
    rows = []
    for keys, g in sales.groupby(["dataset", "design", "config", "timing"], sort=False):
        rows.append(dict(
            dataset=keys[0], design=keys[1], config=keys[2], timing=keys[3],
            **_cell(g["sales"]),
        ))
    return pd.DataFrame(rows)


def table_ii(traj: pd.DataFrame) -> pd.DataFrame:
    sales = _instance_sales(traj)
    if sales.empty:
        return sales
    sales = sales.copy()
    sales["timing"] = _timing(sales)
    sales = sales[sales["timing"] == "early"]
    rows = []
    for keys, g in sales.groupby(["dataset", "design", "config", "method"], sort=False):
        rows.append(dict(
            dataset=keys[0], design=keys[1], config=keys[2], method=keys[3],
            **_cell(g["sales"]),
        ))
    return pd.DataFrame(rows)


def table_iii(buyers: pd.DataFrame) -> pd.DataFrame:
    """Paired excess-loss change vs equal_weight, config=none, plus trials per buyer."""
    hon = _honest(buyers)
    if hon.empty or "excess" not in hon.columns:
        return pd.DataFrame()
    hon = hon[hon["config"] == "none"] if "config" in hon.columns else hon
    inst = (hon.groupby(["dataset", "design", "method", "instance"], sort=False)
            .agg(excess=("excess", "mean"), n_trials=("n_trials", "mean"))
            .reset_index())
    rows = []
    for (ds, design), g in inst.groupby(["dataset", "design"], sort=False):
        base = g[g["method"] == "equal_weight"].set_index("instance")
        if base.empty:
            continue
        for method, m in g.groupby("method", sort=False):
            both = m.set_index("instance").join(base, lsuffix="", rsuffix="_eq", how="inner")
            if both.empty:
                continue
            d = boot_diff(both["excess"].values, both["excess_eq"].values)
            rows.append(dict(
                dataset=ds, design=design, method=method,
                excess_change=d["mean"], excess_lo=d["lo"], excess_hi=d["hi"],
                trials=float(both["n_trials"].mean()),
                trials_equal_weight=float(both["n_trials_eq"].mean()),
                n=d["n"],
            ))
    return pd.DataFrame(rows)


def table_iv(bench: Path) -> pd.DataFrame:
    """Paper Table IV: N, Store, Write(μs), Rank/Thresh./Risk(ms), State(MB)."""
    summary = bench / "table_iv_raw.csv"
    if summary.exists() and summary.stat().st_size > 0:
        df = pd.read_csv(summary)
        keep = [c for c in (
            "N", "Store", "Write_us", "Rank_ms", "Thresh_ms", "Risk_ms", "State_MB",
        ) if c in df.columns]
        return df[keep] if keep else df
    lat_p = bench / "latency.csv"
    write_p = bench / "write_latency.csv"
    state_p = bench / "state_size.csv"
    if not lat_p.exists():
        return pd.DataFrame()
    lat = pd.read_csv(lat_p)
    writes = pd.read_csv(write_p) if write_p.exists() else pd.DataFrame()
    state = pd.read_csv(state_p) if state_p.exists() else pd.DataFrame()
    rows = []
    stores = ["full_recompute", "incremental_exact", "incremental_snapshot"]
    ns = sorted(int(x) for x in lat["n_sellers"].unique()) if len(lat) else []
    for n in ns:
        for store in stores:
            sl = lat[(lat.n_sellers == n) & (lat.store == store)]
            if sl.empty:
                continue
            def _med_ms(q):
                x = sl[sl["query"] == q]["latency_s"]
                return float(np.median(x) * 1000.0) if len(x) else np.nan
            write_us = np.nan
            if len(writes):
                w = writes[(writes.n_sellers == n) & (writes.store == store)]
                if len(w) and "write_s" in w.columns:
                    write_us = float(np.median(w["write_s"]) * 1e6)
                elif len(w) and "write_us" in w.columns:
                    write_us = float(np.median(w["write_us"]))
            state_mb = np.nan
            if len(state):
                st = state[(state.n_sellers == n) & (state.store == store)]
                if len(st) and "state_mb" in st.columns:
                    state_mb = float(st["state_mb"].iloc[0])
            rows.append(dict(
                N=int(n),
                Store=STORE_LABEL.get(store, store),
                Write_us=write_us,
                Rank_ms=_med_ms("ranking"),
                Thresh_ms=_med_ms("threshold"),
                Risk_ms=_med_ms("risk_review"),
                State_MB=state_mb,
            ))
    return pd.DataFrame(rows)


def sensitivity_vs_main(sens: pd.DataFrame) -> pd.DataFrame:
    """Paired target-sales difference vs (Δ=0.05, w_max=0.25), 5000 bootstrap, by instance."""
    if sens.empty:
        return sens
    need = {"dataset", "design", "config", "instance", "w_max", "delta", "packets"}
    if not need.issubset(sens.columns):
        return pd.DataFrame()
    rows = []
    keys = ["dataset", "design", "config"]
    for k, g in sens.groupby(keys, sort=False):
        main = g[(np.abs(g["delta"] - MAIN_DELTA) < 1e-12) & (np.abs(g["w_max"] - MAIN_WMAX) < 1e-12)]
        main = main.drop_duplicates("instance").set_index("instance")["packets"]
        if main.empty:
            continue
        for (delta, w_max), sl in g.groupby(["delta", "w_max"], sort=False):
            sl = sl.drop_duplicates("instance").set_index("instance")["packets"]
            both = pd.concat([sl.rename("alt"), main.rename("main")], axis=1).dropna()
            if both.empty:
                continue
            d = boot_diff(both["alt"].values, both["main"].values)
            n_still = int((sl.values > 0).sum())
            rows.append(dict(
                dataset=k[0], design=k[1], config=k[2],
                delta=float(delta), w_max=float(w_max),
                packets_mean=float(sl.mean()),
                d_packets=d["mean"], d_lo=d["lo"], d_hi=d["hi"],
                n=d["n"], n_still_sold=n_still,
                vs_main=int(abs(float(delta) - MAIN_DELTA) > 1e-12 or abs(float(w_max) - MAIN_WMAX) > 1e-12),
            ))
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in df.itertuples(index=False):
        vals = []
        for v in r:
            if isinstance(v, float) and np.isfinite(v):
                vals.append(f"{v:.3f}" if abs(v) < 10 else f"{v:.2f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _table_i_hit(g: pd.DataFrame, cfg: str) -> pd.DataFrame:
    if cfg == "extreme_late":
        hit = g[(g.config == "extreme") & (_timing(g) == "late")]
    elif cfg == "extreme":
        hit = g[(g.config == "extreme") & (_timing(g) != "late")]
    else:
        hit = g[g.config == cfg]
    return hit


def table_i_wide(t1: pd.DataFrame) -> str:
    if t1.empty:
        return ""
    lines = ["# Table I (equal weight)", ""]
    for ds in ("amazon", "domainnet"):
        sl = t1[t1.dataset == ds]
        if sl.empty:
            continue
        lines.append(f"## {ds}")
        header = ["design"] + [TABLE_I_COL[c] for c in TABLE_I_CONFIGS]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for design, g in sl.groupby("design", sort=False):
            row = [DESIGN_LABEL.get(design, design)]
            for cfg in TABLE_I_CONFIGS:
                hit = _table_i_hit(g, cfg)
                if hit.empty:
                    row.append("--")
                else:
                    r = hit.iloc[0]
                    row.append(_fmt_sales(r.packets_mean, int(r.n_admitted), int(r.n)))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def table_iv_wide(t4: pd.DataFrame) -> str:
    if t4.empty:
        return ""
    lines = [
        "# Table IV",
        "",
        "| N | Store | Write (μs) | Rank (ms) | Thresh. (ms) | Risk (ms) | State (MB) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in t4.itertuples(index=False):
        def _f(v, nd=1):
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                return "--"
            return f"{float(v):.{nd}f}"
        n = int(getattr(r, "N", getattr(r, "n_sellers", 0)))
        store = str(getattr(r, "Store", getattr(r, "store", "")))
        lines.append(
            f"| {n} | {store} | {_f(getattr(r, 'Write_us', np.nan), 1)} | "
            f"{_f(getattr(r, 'Rank_ms', np.nan), 3)} | "
            f"{_f(getattr(r, 'Thresh_ms', np.nan), 3)} | "
            f"{_f(getattr(r, 'Risk_ms', np.nan), 3)} | "
            f"{_f(getattr(r, 'State_MB', np.nan), 1)} |"
        )
    return "\n".join(lines) + "\n"


METHOD_LABEL = {
    "equal_weight": "Equal Weight",
    "value_filter": "Value Filter",
    "median": "Median",
    "trimmed_mean": "Trimmed Mean",
    "purchase_weight": "Purchase Weight",
    "forced_trials": "Forced Repeat Trials",
    "qsource": "QSource",
}
DESIGN_LABEL = {
    "no_sharing": "No Sharing",
    "row_level": "Row-Level Sharing",
    "buyer_level": "Buyer-Level Sharing",
    "buyer_level_floor": "Buyer-Level Sharing + Floor",
    "known_quality": "Known Reliability",
}


def table_ii_wide(t2: pd.DataFrame) -> str:
    if t2.empty:
        return ""
    methods = [
        "equal_weight", "purchase_weight", "value_filter", "forced_trials",
        "qsource", "median", "trimmed_mean",
    ]
    lines = ["# Table II", ""]
    for ds in ("amazon", "domainnet"):
        sl = t2[t2.dataset == ds]
        if sl.empty:
            continue
        lines.append(f"## {ds}")
        header = ["design", "config"] + [METHOD_LABEL.get(m, m) for m in methods]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for (design, config), g in sl.groupby(["design", "config"], sort=False):
            row = [DESIGN_LABEL.get(design, design), config]
            for m in methods:
                hit = g[g.method == m]
                if hit.empty:
                    row.append("--")
                else:
                    r = hit.iloc[0]
                    row.append(_fmt_sales(r.packets_mean, int(r.n_admitted), int(r.n)))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


FLOOR_METHODS = ("equal_weight", "qsource")


def _read_meta(root: Path) -> pd.DataFrame:
    rows = []
    top = root / "meta.json"
    if top.exists() and top.stat().st_size > 0:
        blob = json.loads(top.read_text())
        rows.extend(blob if isinstance(blob, list) else [blob])
    cells = root / "cells"
    if cells.is_dir():
        for p in sorted(cells.glob("*/meta.json")):
            try:
                rows.append(json.loads(p.read_text()))
            except json.JSONDecodeError:
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    keys = [c for c in list(CELL) + ["skip_threshold", "target_goal"] if c in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys, keep="last")
    return df


def _floor_one(plat, pays, target) -> dict:
    plat_spend = float(plat["platform_paid"].sum()) if len(plat) and "platform_paid" in plat.columns else 0.0
    n_plat = int(len(plat))
    if target is not None and len(plat) and "seller" in plat.columns:
        hit = plat[plat["seller"].astype(int) == int(target)]
        plat_tgt = float(hit["platform_paid"].sum()) if len(hit) and "platform_paid" in hit.columns else 0.0
        n_plat_tgt = int(len(hit))
    else:
        plat_tgt, n_plat_tgt = 0.0, 0
    hon = _honest(pays)
    pkt = float(hon["packet_paid"].sum()) if len(hon) and "packet_paid" in hon.columns else 0.0
    ratio = plat_spend / pkt if pkt > 1e-15 else np.nan
    return dict(
        n_plat_trials=n_plat,
        n_plat_trials_target=n_plat_tgt,
        plat_spend=plat_spend,
        plat_spend_target=plat_tgt,
        hon_packet_spend=pkt,
        plat_over_packet=ratio,
        target_seller=target,
    )


def floor_cost(plat: pd.DataFrame, pays: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Platform floor spend vs honest packet spend on buyer_level_floor.

    Per cell: equal_weight and qsource, both markets, none and every injection
    config. Also qsource − equal_weight paired differences.
    """
    if plat is None or plat.empty or pays is None or pays.empty:
        return pd.DataFrame(), pd.DataFrame()
    plat = plat.copy()
    pays = pays.copy()
    if "skip_threshold" in plat.columns:
        plat = plat[np.abs(plat["skip_threshold"].fillna(1.0).astype(float) - 1.0) < 1e-12]
    if "skip_threshold" in pays.columns:
        pays = pays[np.abs(pays["skip_threshold"].fillna(1.0).astype(float) - 1.0) < 1e-12]
    if "target_goal" in plat.columns:
        plat = plat[plat["target_goal"].fillna("lower").astype(str).isin(("lower", ""))]
    if "target_goal" in pays.columns:
        pays = pays[pays["target_goal"].fillna("lower").astype(str).isin(("lower", ""))]
    if "design" in plat.columns:
        plat = plat[plat["design"].astype(str) == "buyer_level_floor"]
    if "method" in plat.columns:
        plat = plat[plat["method"].astype(str).isin(FLOOR_METHODS)]
    if "design" in pays.columns:
        pays = pays[pays["design"].astype(str) == "buyer_level_floor"]
    if "method" in pays.columns:
        pays = pays[pays["method"].astype(str).isin(FLOOR_METHODS)]
    keys = [c for c in ("dataset", "design", "method", "config", "timing", "instance") if c in plat.columns]
    tgt_map = {}
    if meta is not None and len(meta):
        mk = [c for c in keys if c in meta.columns]
        for r in meta.itertuples(index=False):
            key = tuple(str(getattr(r, c)) if c != "instance" else int(getattr(r, c)) for c in mk)
            tgt_map[key] = getattr(r, "target_seller", None)

    rows = []
    for k, gplat in plat.groupby(keys, sort=False):
        rec = {keys[i]: k[i] for i in range(len(keys))}
        gpays = pays
        for c in keys:
            if c in gpays.columns:
                gpays = gpays[gpays[c].astype(str) == str(rec[c])]
        key = tuple(str(rec[c]) if c != "instance" else int(rec[c]) for c in keys if c in (meta.columns if meta is not None and len(meta) else []))
        tgt = tgt_map.get(key)
        if tgt is None and rec.get("config") not in (None, "none"):
            # injection always has a target in meta; if missing, skip target split
            tgt = None
        try:
            tgt = None if tgt is None or (isinstance(tgt, float) and not np.isfinite(tgt)) else int(tgt)
        except (TypeError, ValueError):
            tgt = None
        rec.update(_floor_one(gplat, gpays, tgt))
        rows.append(rec)
    inst = pd.DataFrame(rows)
    diff_rows = []
    if inst.empty:
        return inst, pd.DataFrame()
    pair_keys = [c for c in ("dataset", "design", "config", "timing", "instance") if c in inst.columns]
    qs = inst[inst["method"] == "qsource"]
    ew = inst[inst["method"] == "equal_weight"]
    if len(qs) and len(ew):
        j = qs.merge(ew, on=pair_keys, suffixes=("_qsource", "_equal_weight"))
        for _, r in j.iterrows():
            rec = {c: r[c] for c in pair_keys}
            rec["plat_spend_diff"] = float(r["plat_spend_qsource"] - r["plat_spend_equal_weight"])
            rec["plat_spend_target_diff"] = float(
                r["plat_spend_target_qsource"] - r["plat_spend_target_equal_weight"]
            )
            rec["hon_packet_spend_diff"] = float(
                r["hon_packet_spend_qsource"] - r["hon_packet_spend_equal_weight"]
            )
            diff_rows.append(rec)
    return inst, pd.DataFrame(diff_rows)


def floor_cost_md(inst: pd.DataFrame, diffs: pd.DataFrame) -> str:
    lines = [
        "# Floor cost",
        "",
        "Platform floor spend (packet-price units), the part spent on the target seller, "
        "honest-buyer packet spend, and their ratio. Buyer-level + floor only.",
        "",
    ]
    if len(inst):
        lines.append(to_markdown(inst))
        lines.append("")
    if len(diffs):
        lines.append("## qsource − equal_weight")
        lines.append("")
        lines.append(to_markdown(diffs))
        lines.append("")
    return "\n".join(lines)


def _skip_k(df: pd.DataFrame) -> pd.Series:
    if "skip_threshold" in df.columns:
        return df["skip_threshold"].fillna(1.0).astype(float)
    return pd.Series(1.0, index=df.index)


def _goal(df: pd.DataFrame) -> pd.Series:
    if "target_goal" in df.columns:
        g = df["target_goal"].fillna("lower").astype(str)
        return g.where(g != "", "lower")
    return pd.Series("lower", index=df.index)


def _trials_col(df: pd.DataFrame) -> str | None:
    if "n_trials" in df.columns:
        return "n_trials"
    if "n_checked" in df.columns:
        return "n_checked"
    return None


def _join_extra(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if meta is None or meta.empty:
        if "skip_threshold" not in out.columns:
            out["skip_threshold"] = 1.0
        if "target_goal" not in out.columns:
            out["target_goal"] = "lower"
        return out
    keys = [c for c in CELL if c in out.columns and c in meta.columns]
    extra = [c for c in ("skip_threshold", "target_goal", "target_seller") if c in meta.columns]
    if keys and extra:
        add = meta[keys + extra].drop_duplicates(subset=keys, keep="last")
        missing = [c for c in extra if c not in out.columns]
        if missing:
            out = out.merge(add[keys + missing], on=keys, how="left")
    if "skip_threshold" not in out.columns:
        out["skip_threshold"] = 1.0
    else:
        out["skip_threshold"] = out["skip_threshold"].fillna(1.0).astype(float)
    if "target_goal" not in out.columns:
        out["target_goal"] = "lower"
    else:
        g = out["target_goal"].fillna("lower").astype(str)
        out["target_goal"] = g.where(g != "", "lower")
    return out


def _instance_sales_any(
    traj: pd.DataFrame, pays: pd.DataFrame, meta: pd.DataFrame,
) -> pd.DataFrame:
    """Target-seller packets per instance. Trajectories first; else pays × meta target."""
    sales = _instance_sales(traj) if traj is not None and len(traj) else pd.DataFrame()
    if len(sales):
        return _join_extra(sales, meta)
    if pays is None or pays.empty or meta is None or meta.empty:
        return pd.DataFrame()
    if "target_seller" not in meta.columns or "packets_bought" not in pays.columns:
        return pd.DataFrame()
    hon = _honest(pays)
    hon = _join_extra(hon, meta)
    mk = [c for c in CELL if c in hon.columns and c in meta.columns]
    if not mk or "target_seller" not in hon.columns:
        return pd.DataFrame()
    hon = hon[hon["target_seller"].notna()]
    if hon.empty:
        return pd.DataFrame()
    hon = hon[hon["seller"].astype(int) == hon["target_seller"].astype(int)]
    keys = [c for c in CELL + ["skip_threshold", "target_goal"] if c in hon.columns]
    return hon.groupby(keys, sort=False)["packets_bought"].sum().reset_index(name="sales")


def _instance_buyers(buyers: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    hon = _honest(buyers)
    if hon.empty or "excess" not in hon.columns:
        return pd.DataFrame()
    hon = _join_extra(hon, meta)
    tc = _trials_col(hon)
    keys = [c for c in CELL + ["skip_threshold", "target_goal"] if c in hon.columns]
    agg = dict(excess=("excess", "mean"))
    if tc:
        agg["n_trials"] = (tc, "mean")
    return hon.groupby(keys, sort=False).agg(**agg).reset_index()


def _boot_pair(a: pd.Series, b: pd.Series) -> dict:
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if both.empty:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, n=0)
    return boot_diff(both["a"].values, both["b"].values)


def additional_experiments(
    traj: pd.DataFrame, buyers: pd.DataFrame,
    pays: pd.DataFrame, meta: pd.DataFrame,
) -> pd.DataFrame:
    """raise_weak vs none, skip 2 vs 1, Purchase Weight vs QSource.

    All rows are instance-paired, 5000-bootstrap 95% intervals.
    """
    sales = _instance_sales_any(traj, pays, meta)
    inst = _instance_buyers(buyers, meta)
    rows = []

    def _add(comparison, dataset, method, config, metric, d):
        n = int(d.get("n") or 0)
        if n <= 0:
            rows.append(dict(
                comparison=comparison, dataset=dataset, method=method,
                config=config, metric=metric,
                mean="NA", lo="NA", hi="NA", n=0,
            ))
            return
        rows.append(dict(
            comparison=comparison, dataset=dataset, method=method,
            config=config, metric=metric,
            mean=d.get("mean", np.nan), lo=d.get("lo", np.nan),
            hi=d.get("hi", np.nan), n=n,
        ))

    # raise_weak vs same-target none (target_goal=raise), skip=1
    if len(sales):
        sl = sales.copy()
        sl = sl[sl["design"].astype(str) == "buyer_level_floor"] if "design" in sl.columns else sl
        sl = sl[np.abs(_skip_k(sl) - 1.0) < 1e-12]
        sl = sl[sl["method"].astype(str).isin(("equal_weight", "qsource"))] if "method" in sl.columns else sl
        treat = sl[
            (sl["config"].astype(str) == "raise_weak")
            | ((sl["config"].astype(str) == "none") & (_goal(sl) == "raise"))
        ]
        for (ds, method), g in treat.groupby(["dataset", "method"], sort=False):
            rw = g[g["config"].astype(str) == "raise_weak"].drop_duplicates("instance").set_index("instance")["sales"]
            none = g[(g["config"].astype(str) == "none") & (_goal(g) == "raise")]
            none = none.drop_duplicates("instance").set_index("instance")["sales"]
            _add("raise_weak_vs_none", ds, method, "raise_weak", "sales", _boot_pair(rw, none))
    if len(inst):
        bi = inst.copy()
        bi = bi[bi["design"].astype(str) == "buyer_level_floor"] if "design" in bi.columns else bi
        bi = bi[np.abs(_skip_k(bi) - 1.0) < 1e-12]
        bi = bi[bi["method"].astype(str).isin(("equal_weight", "qsource"))] if "method" in bi.columns else bi
        treat = bi[
            (bi["config"].astype(str) == "raise_weak")
            | ((bi["config"].astype(str) == "none") & (_goal(bi) == "raise"))
        ]
        for (ds, method), g in treat.groupby(["dataset", "method"], sort=False):
            rw = g[g["config"].astype(str) == "raise_weak"].drop_duplicates("instance").set_index("instance")["excess"]
            none = g[(g["config"].astype(str) == "none") & (_goal(g) == "raise")]
            none = none.drop_duplicates("instance").set_index("instance")["excess"]
            _add("raise_weak_vs_none", ds, method, "raise_weak", "excess", _boot_pair(rw, none))

    # skip-threshold 2 vs 1
    for frame, metric, col in (
        (sales, "sales", "sales"),
        (inst, "excess", "excess"),
        (inst, "n_trials", "n_trials"),
    ):
        if frame is None or frame.empty or col not in frame.columns:
            continue
        sl = frame.copy()
        sl = sl[sl["design"].astype(str) == "buyer_level_floor"] if "design" in sl.columns else sl
        sl = sl[sl["method"].astype(str).isin(("equal_weight", "qsource"))] if "method" in sl.columns else sl
        sl = sl[_goal(sl) != "raise"]
        sl = sl[sl["config"].astype(str) != "raise_weak"]
        for (ds, method, cfg), g in sl.groupby(["dataset", "method", "config"], sort=False):
            a = g[np.abs(_skip_k(g) - 2.0) < 1e-12].drop_duplicates("instance").set_index("instance")[col]
            b = g[np.abs(_skip_k(g) - 1.0) < 1e-12].drop_duplicates("instance").set_index("instance")[col]
            _add("skip2_vs_skip1", ds, method, cfg, metric, _boot_pair(a, b))

    # Purchase Weight − QSource excess on buyer_level_floor
    if len(inst) and "method" in inst.columns:
        bi = inst.copy()
        bi = bi[bi["design"].astype(str) == "buyer_level_floor"]
        bi = bi[np.abs(_skip_k(bi) - 1.0) < 1e-12]
        bi = bi[_goal(bi) != "raise"]
        bi = bi[bi["config"].astype(str) != "raise_weak"]
        for (ds, cfg), g in bi.groupby(["dataset", "config"], sort=False):
            pw = g[g["method"] == "purchase_weight"].drop_duplicates("instance").set_index("instance")["excess"]
            qs = g[g["method"] == "qsource"].drop_duplicates("instance").set_index("instance")["excess"]
            _add("purchase_weight_vs_qsource", ds, "purchase_weight", cfg, "excess", _boot_pair(pw, qs))

    return pd.DataFrame(rows)


def additional_experiments_md(df: pd.DataFrame) -> str:
    lines = [
        "# Additional experiments",
        "",
        "Instance-paired differences, 5000 bootstrap, 95% interval.",
        "",
        "- `raise_weak_vs_none`: raise_weak minus same-target none (`target_goal=raise`).",
        "- `skip2_vs_skip1`: skip-threshold 2 minus 1.",
        "- `purchase_weight_vs_qsource`: Purchase Weight minus QSource excess, buyer_level_floor.",
        "",
    ]
    if df is None or df.empty:
        lines.append("(no paired cells)")
        lines.append("")
        return "\n".join(lines)
    lines.append(to_markdown(df))
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    root = Path(args.results)
    out = Path(args.out) if args.out else root
    out.mkdir(parents=True, exist_ok=True)
    traj = _read_all(root, "trajectories.csv")
    if traj.empty:
        for alt in (root / "sensitivity",):
            t = _read_all(alt, "trajectories.csv")
            if len(t):
                traj = pd.concat([traj, t], ignore_index=True) if len(traj) else t
    if len(traj):
        t1 = table_i(traj)
        t1.to_csv(out / "table_i.csv", index=False)
        (out / "table_i.md").write_text(table_i_wide(t1) or to_markdown(t1))
        t2 = table_ii(traj)
        t2.to_csv(out / "table_ii.csv", index=False)
        (out / "table_ii.md").write_text(table_ii_wide(t2) or to_markdown(t2))
        print("table I", len(t1), "table II", len(t2))
    buy = _read_all(root, "buyers.csv")
    if len(buy):
        t3 = table_iii(buy)
        if len(t3):
            t3.to_csv(out / "table_iii.csv", index=False)
            (out / "table_iii.md").write_text(to_markdown(t3))
            print("table III", len(t3))
    bench = root / "benchmark"
    t4 = table_iv(bench)
    if len(t4):
        t4.to_csv(out / "table_iv.csv", index=False)
        (out / "table_iv.md").write_text(table_iv_wide(t4) or to_markdown(t4))
        print("table IV", len(t4))
    sens_p = root / "sensitivity" / "sensitivity.csv"
    if not sens_p.exists():
        sens_p = root / "sensitivity.csv"
    if sens_p.exists():
        sens = pd.read_csv(sens_p)
        vs = sensitivity_vs_main(sens)
        vs.to_csv(out / "sensitivity.csv", index=False)
        (out / "sensitivity.md").write_text(to_markdown(vs))
        print("sensitivity", len(vs))
    plat = _read_all(root, "plat.csv")
    pays = _read_all(root, "pays.csv")
    meta = _read_meta(root)
    if len(plat) and len(pays):
        fc, fd = floor_cost(plat, pays, meta)
        if len(fc):
            fc.to_csv(out / "floor_cost.csv", index=False)
            if len(fd):
                fd.to_csv(out / "floor_cost_diff.csv", index=False)
            (out / "floor_cost.md").write_text(floor_cost_md(fc, fd))
            print("floor cost", len(fc), "paired diffs", len(fd))
    add = additional_experiments(traj, buy, pays, meta)
    if len(add):
        add.to_csv(out / "additional_experiments.csv", index=False)
        (out / "additional_experiments.md").write_text(additional_experiments_md(add))
        print("additional experiments", len(add))


if __name__ == "__main__":
    main()
