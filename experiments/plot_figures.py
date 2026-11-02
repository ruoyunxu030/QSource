"""Plot the figures of the paper from the CSV files written by the experiment scripts.

    python plot_figures.py --results results --out figures

Inputs (all under --results):
  trajectories.csv           one row per (dataset, design, config, method, instance, buyer)
      dataset   amazon | domainnet
      design    row_level | buyer_level | buyer_level_floor
      config    none | low_honest | mediocre | fixed | extreme | extreme_one_account
      method    equal_weight | value_filter | forced_trials | qsource
      instance  instance id
      buyer     honest buyer t = 1..30
      packets   packets of the target bought by buyer t
      published published estimate of the target that buyer t reads
  benchmark/latency.csv      n_sellers, store, query, latency_s
      store     full_recompute | incremental_exact | incremental_snapshot
      query     ranking | threshold | risk_review
  benchmark/snapshot_error.csv   n_sellers, buyer, refresh, mean_abs_error
      refresh   1 at the buyers where the snapshot is refreshed, else 0

Outputs: fig2_published_estimate.pdf, fig4_target_sales.pdf, fig5_maintenance.pdf
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Carlito", "Calibri", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "grid.color": "#E3E3E3",
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.25,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})

METHODS = {  # label, color, line style
    "none": ("No Injection", "#70AD47", (0, (1.2, 1.2))),
    "equal_weight": ("Equal Weight", "#8C8C8C", (0, (4, 2))),
    "value_filter": ("Value Filter", "#5FA8A0", "-"),
    "forced_trials": ("Forced Repeat Trials", "#E0803C", "-"),
    "qsource": ("QSource", "#1F2D3D", "-"),
}
DESIGNS = {"buyer_level": ("Buyer-Level Sharing", "#7A5EA8"),
           "buyer_level_floor": ("Buyer-Level Sharing + Floor", "#2F5597")}
STORES = {"full_recompute": ("Full Recompute", "#8C8C8C", "s"),
          "incremental_exact": ("Incremental Exact", "#1F2D3D", "o"),
          "incremental_snapshot": ("Incremental Snapshot", "#5FA8A0", "^")}
CONFIG_LABEL = {"low_honest": "Low Honest", "mediocre": "Mediocre", "fixed": "Fixed 0.5",
                "extreme": "Extreme", "extreme_one_account": "Extreme, 1 Account"}
DATASET_LABEL = {"amazon": "Amazon", "domainnet": "DomainNet"}


def bootstrap_band(values, n_boot=5000, seed=0):
    """95% band of the mean over instances; values has shape (instances, buyers)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return np.percentile(means, 2.5, axis=0), np.percentile(means, 97.5, axis=0)


def per_buyer(df, column):
    """Matrix instances x buyers of one column, buyers sorted."""
    return df.pivot_table(index="instance", columns="buyer", values=column, aggfunc="mean").sort_index(axis=1)


def buyer_axis(ax):
    ax.set_xlim(1, 30)
    ax.set_xticks([1, 10, 20, 30])
    ax.set_xlabel("Honest buyer $t$")


def legend_below(fig, handles, ncol):
    fig.legend(handles=handles, loc="lower center", ncol=ncol, handlelength=2.2, columnspacing=1.4)


def fig2_published_estimate(traj, out):
    """Published estimate of the target under Equal Weight, with and without Extreme records."""
    fig, axes = plt.subplots(1, 2, figsize=(3.6, 1.9))
    for ax, dataset in zip(axes, ["amazon", "domainnet"]):
        for design, (_, color) in DESIGNS.items():
            for config, style in [("none", "-"), ("extreme", (0, (4, 1.6)))]:
                g = traj[(traj.dataset == dataset) & (traj.design == design)
                         & (traj.method == "equal_weight") & (traj.config == config)]
                if g.empty:
                    continue
                m = per_buyer(g, "published")
                ax.plot(m.columns, m.mean(axis=0), color=color, ls=style)
        buyer_axis(ax)
        ax.set_ylim(0, 1.02)
        ax.set_title(DATASET_LABEL[dataset])
    axes[0].set_ylabel("Published estimate of target")
    handles = [plt.Line2D([], [], color=c, label=l) for l, c in DESIGNS.values()]
    handles.append(plt.Line2D([], [], color="#555555", ls=(0, (4, 1.6)), label="with Extreme records"))
    legend_below(fig, handles, ncol=2)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.88, bottom=0.42, wspace=0.3)
    fig.savefig(os.path.join(out, "fig2_published_estimate.pdf"))
    plt.close(fig)


def fig4_target_sales(traj, out):
    """Cumulative packet sales of the target with the floor, mean over instances with a 95% band."""
    panels = [("amazon", "low_honest"), ("amazon", "extreme"), ("domainnet", "fixed"), ("domainnet", "extreme")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.9))
    floor = traj[traj.design == "buyer_level_floor"]
    for ax, (dataset, config) in zip(axes, panels):
        d = floor[floor.dataset == dataset]
        curves = {"none": d[(d.config == "none") & (d.method == "qsource")]}
        for method in ["equal_weight", "value_filter", "forced_trials", "qsource"]:
            curves[method] = d[(d.config == config) & (d.method == method)]
        for method, g in curves.items():
            if g.empty:
                continue
            cum = per_buyer(g, "packets").fillna(0).cumsum(axis=1)
            lo, hi = bootstrap_band(cum.values)
            _, color, style = METHODS[method]
            ax.fill_between(cum.columns, lo, hi, color=color, alpha=0.10, lw=0)
            ax.plot(cum.columns, cum.mean(axis=0), color=color, ls=style, zorder=3 if method == "qsource" else 2)
        buyer_axis(ax)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{DATASET_LABEL[dataset]}, {CONFIG_LABEL[config]}")
    axes[0].set_ylabel("Cumulative target sales")
    handles = [plt.Line2D([], [], color=c, ls=s, label=l) for l, c, s in METHODS.values()]
    legend_below(fig, handles, ncol=5)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.33, wspace=0.32)
    fig.savefig(os.path.join(out, "fig4_target_sales.pdf"))
    plt.close(fig)


def fig5_maintenance(bench_dir, out):
    """Query latency against the number of sellers, and the error of the snapshot store between refreshes."""
    lat = pd.read_csv(os.path.join(bench_dir, "latency.csv"))
    err = pd.read_csv(os.path.join(bench_dir, "snapshot_error.csv"))
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.9))
    titles = {"ranking": "Ranking query", "threshold": "Threshold query", "risk_review": "Risk-review query"}
    sizes = sorted(lat.n_sellers.unique())
    for ax, (query, title) in zip(axes, titles.items()):
        for store, (_, color, marker) in STORES.items():
            g = lat[(lat["query"] == query) & (lat.store == store)].groupby("n_sellers").latency_s
            if g.ngroups == 0:
                continue
            ax.fill_between(sizes, g.quantile(0.05).reindex(sizes) * 1e3, g.quantile(0.95).reindex(sizes) * 1e3,
                            color=color, alpha=0.18, lw=0)
            ax.plot(sizes, g.median().reindex(sizes) * 1e3, color=color, marker=marker, ms=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mt.LogLocator(base=10, numticks=8))
        ax.yaxis.set_minor_locator(mt.NullLocator())
        ax.set_xticks(sizes)
        ax.set_xlabel("Number of sellers $N$")
        ax.set_title(title)
    axes[0].set_ylabel("Latency (ms)")

    ax = axes[3]
    e = err[err.n_sellers == 10_000].sort_values("buyer")
    for b in e.loc[e.refresh == 1, "buyer"]:
        ax.axvline(b / 1e3, color="#E3B75A", lw=0.7, ls=(0, (1.5, 1.5)))
    color = STORES["incremental_snapshot"][1]
    ax.fill_between(e.buyer / 1e3, 0, e.mean_abs_error * 1e3, color=color, alpha=0.15, lw=0)
    ax.plot(e.buyer / 1e3, e.mean_abs_error * 1e3, color=color, marker="^", ms=2.6, lw=1.0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Buyers processed ($\\times 10^3$)")
    ax.set_ylabel("Mean abs. error ($\\times 10^{-3}$)")
    ax.set_title("Snapshot error, $N=10^4$")

    handles = [plt.Line2D([], [], color=c, marker=m, ms=3, label=l) for l, c, m in STORES.values()]
    legend_below(fig, handles, ncol=3)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.33, wspace=0.42)
    fig.savefig(os.path.join(out, "fig5_maintenance.pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    traj = pd.read_csv(os.path.join(args.results, "trajectories.csv"))
    fig2_published_estimate(traj, args.out)
    fig4_target_sales(traj, args.out)
    bench = os.path.join(args.results, "benchmark")
    if os.path.exists(os.path.join(bench, "latency.csv")) and os.path.exists(
        os.path.join(bench, "snapshot_error.csv")
    ):
        fig5_maintenance(bench, args.out)
    else:
        print("skip fig5: missing benchmark/latency.csv or snapshot_error.csv")


if __name__ == "__main__":
    main()
