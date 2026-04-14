"""
analyze_results.py

Aggregates results from run_padding_benchmark.py (JSON files in results/)
and produces:

  1. A summary CSV with mean metrics per (config, pad%, CoV) cell.
  2. A 4×4 heatmap of effective throughput (tokens/sec) for each config.
  3. A line plot: effective MFU proxy vs. pad% faceted by config.

Usage:
    python scripts/analyze_results.py --results-dir results --plots-dir plots
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAD_LEVELS = [0.20, 0.30, 0.50, 0.70]
COV_LEVELS = [0.05, 0.20, 0.35, 0.50]
CONFIG_LABELS = {
    "config_a": "Config A (TP=4, PP=1)",
    "config_b": "Config B (TP=2, PP=2)",
    "config_c": "Config C (TP=4, PP=1+SP)",
}


def load_all_results(results_dir: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_results.json"))):
        with open(path) as f:
            data = json.load(f)
        if data:
            frames.append(pd.DataFrame(data))
    if not frames:
        raise FileNotFoundError(f"No *_results.json files found in {results_dir}")
    return pd.concat(frames, ignore_index=True)


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["config", "target_pad_ratio", "target_cov"])
        .agg(
            mean_wall_clock_s=("wall_clock_s", "mean"),
            mean_raw_tps=("raw_tps", "mean"),
            mean_effective_tps=("effective_tps", "mean"),
            mean_max_memory_gb=("max_memory_gb", "mean"),
            n_batches=("batch_id", "count"),
        )
        .reset_index()
    )
    return agg


def plot_heatmaps(summary: pd.DataFrame, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)
    configs = summary["config"].unique()

    for config in configs:
        cfg_df = summary[summary["config"] == config]
        grid = np.full((len(COV_LEVELS), len(PAD_LEVELS)), np.nan)

        for ri, cov in enumerate(COV_LEVELS):
            for ci, pad in enumerate(PAD_LEVELS):
                row = cfg_df[
                    (cfg_df["target_pad_ratio"].round(2) == round(pad, 2))
                    & (cfg_df["target_cov"].round(2) == round(cov, 2))
                ]
                if not row.empty:
                    grid[ri, ci] = row["mean_effective_tps"].values[0]

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(grid, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(PAD_LEVELS)))
        ax.set_xticklabels([f"{int(p * 100)}%" for p in PAD_LEVELS], fontsize=10)
        ax.set_yticks(range(len(COV_LEVELS)))
        ax.set_yticklabels([f"{c:.2f}" for c in COV_LEVELS], fontsize=10)
        ax.set_xlabel("Target pad%", fontsize=11)
        ax.set_ylabel("Target CoV", fontsize=11)
        label = CONFIG_LABELS.get(config, config)
        ax.set_title(f"Effective throughput (tokens/s)\n{label}", fontsize=12)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("tok/s", fontsize=9)

        for ri in range(len(COV_LEVELS)):
            for ci in range(len(PAD_LEVELS)):
                val = grid[ri, ci]
                if not np.isnan(val):
                    ax.text(ci, ri, f"{val:.0f}", ha="center", va="center",
                            fontsize=8, color="white" if val < grid.max() * 0.6 else "black")

        plt.tight_layout()
        path = os.path.join(plots_dir, f"{config}_heatmap.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved heatmap → {path}")


def plot_pad_vs_throughput(summary: pd.DataFrame, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)
    configs = sorted(summary["config"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: effective tps vs pad%
    for config in configs:
        cfg_df = summary[summary["config"] == config]
        pad_means = (
            cfg_df.groupby("target_pad_ratio")["mean_effective_tps"]
            .mean()
            .reset_index()
            .sort_values("target_pad_ratio")
        )
        axes[0].plot(
            pad_means["target_pad_ratio"] * 100,
            pad_means["mean_effective_tps"],
            marker="o",
            label=CONFIG_LABELS.get(config, config),
        )
    axes[0].set_xlabel("Padding %", fontsize=11)
    axes[0].set_ylabel("Effective throughput (tok/s)", fontsize=11)
    axes[0].set_title("Effective throughput vs. padding %\n(averaged over CoV levels)", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Right: raw tps vs pad% (to show overhead)
    for config in configs:
        cfg_df = summary[summary["config"] == config]
        pad_means = (
            cfg_df.groupby("target_pad_ratio")["mean_raw_tps"]
            .mean()
            .reset_index()
            .sort_values("target_pad_ratio")
        )
        axes[1].plot(
            pad_means["target_pad_ratio"] * 100,
            pad_means["mean_raw_tps"],
            marker="s",
            linestyle="--",
            label=CONFIG_LABELS.get(config, config),
        )
    axes[1].set_xlabel("Padding %", fontsize=11)
    axes[1].set_ylabel("Raw throughput (tok/s)", fontsize=11)
    axes[1].set_title("Raw throughput vs. padding %\n(includes padding tokens)", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, "throughput_vs_padding.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved throughput plot → {path}")


def print_pivot_table(summary: pd.DataFrame):
    print("\n" + "=" * 72)
    print("MEAN EFFECTIVE THROUGHPUT (tok/s) — rows=CoV target, cols=pad%")
    print("=" * 72)
    for config in sorted(summary["config"].unique()):
        cfg_df = summary[summary["config"] == config]
        pivot = cfg_df.pivot_table(
            values="mean_effective_tps",
            index="target_cov",
            columns="target_pad_ratio",
            aggfunc="mean",
        ).reindex(index=COV_LEVELS, columns=PAD_LEVELS)
        pivot.columns = [f"pad={int(p * 100)}%" for p in pivot.columns]
        pivot.index = [f"CoV={c:.2f}" for c in pivot.index]
        print(f"\n{CONFIG_LABELS.get(config, config)}")
        print(pivot.to_string(float_format="{:.1f}".format))
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Aggregate padding benchmark results")
    parser.add_argument("--results-dir", default="results", help="Directory with *_results.json")
    parser.add_argument("--plots-dir", default="plots", help="Output directory for plots")
    parser.add_argument("--summary-csv", default=None, help="Optional path for summary CSV output")
    args = parser.parse_args()

    df = load_all_results(args.results_dir)
    print(f"Loaded {len(df)} batch records from {args.results_dir}")

    summary = make_summary(df)

    summary_csv = args.summary_csv or os.path.join(args.results_dir, "summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Summary CSV → {summary_csv}")

    plot_heatmaps(summary, args.plots_dir)
    plot_pad_vs_throughput(summary, args.plots_dir)
    print_pivot_table(summary)


if __name__ == "__main__":
    main()
