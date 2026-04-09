"""
preprocess_dataset.py

Prepares the ShareGPT dataset for the 4×4 controlled padding experiment.

Experiment design (from project proposal)
------------------------------------------
The full experiment is a grid of N × 4 × 4 trials, where N is the number of
parallelism configurations (TP/PP combos).  The two padding dimensions are:

  pad_ratio  — fraction of token slots in a batch that are padding:
                  pad% = (bs × max_len − Σ lengths) / (bs × max_len)
               Four levels: 10%, 30%, 50%, 70%  (proposal §2.2)

  CoV        — coefficient of variation of sequence lengths within a batch:
                  CoV = std(lengths) / mean(lengths)
               Controls how spread-out lengths are for a fixed pad%.
               Two batches with identical pad% can differ substantially
               in how that padding is distributed across sequences.
               Four levels: 0.05, 0.20, 0.40, 0.65

Budget note
-----------
3 parallelism configs × 16 cells × 5 batches × ~37 s/batch ≈ 2.47 h ≈ 474 SU.
This fits within the 480 SU (2.5 h) requested in the proposal.
N_BATCHES_PER_CELL = 5 matches the "~5 batches per trial" in the proposal.

Pipeline
--------
  1. Load ShareGPT from HuggingFace, extract first human turn.
  2. Tokenize with the Qwen3-32B tokenizer.
  3. Analyse and plot the length distribution.
  4. Filter to [32, 1024] tokens.
  5. Build 16-cell benchmark with joint pad% + CoV control.
  6. Save one JSON per cell + summary CSV.
  7. Produce 4×4 heatmap plots and print a summary pivot table.

To run:
    pip install transformers datasets matplotlib numpy pandas
    python scripts/preprocess_dataset.py 2>&1 | tee run.log

Output structure:
    benchmark_dataset/
      pad10_cov005.json   # cell: 10% padding, CoV=0.05  (5 batches of 64)
      pad10_cov020.json
      ...                 # 16 files total
      benchmark_summary.csv

    plots/
      length_distribution.png
      benchmark_grid.png
"""

import json
import os
import random
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")          # headless — safe on GPU servers without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_NAME   = "anon8231489123/ShareGPT_Vicuna_unfiltered"
TOKENIZER_NAME = "Qwen/Qwen3-32B"

MIN_TOKENS = 32
MAX_TOKENS = 1024

BATCH_SIZE = 64

# Padding ratio levels (proposal §2.2: "10%, 30%, 50%, 70%")
PAD_LEVELS = [0.10, 0.30, 0.50, 0.70]

# Intra-batch CoV levels: std(fill_lengths) / mean(fill_lengths).
# Chosen to be achievable across all pad levels given ShareGPT's distribution.
#   0.05 → near-uniform lengths (all sequences similar length)
#   0.20 → mild spread
#   0.40 → moderate spread
#   0.65 → high spread (mix of short and longer fills)
COV_LEVELS = [0.05, 0.20, 0.40, 0.65]

# Batches per (pad, CoV) cell.
# Budget: 3 configs × 16 cells × 5 batches × ~37 s ≈ 474 SU  (within 480 SU)
N_BATCHES_PER_CELL = 5

OUTPUT_DIR = "benchmark_dataset"
PLOTS_DIR  = "plots"
SEED = 42


# ---------------------------------------------------------------------------
# 1. Load dataset and extract first-turn prompts
# ---------------------------------------------------------------------------

def load_sharegpt_prompts(dataset_name: str) -> List[str]:
    """
    Download ShareGPT and return the first human turn of every conversation.

    ShareGPT conversations are stored as a list of dicts under the key
    "conversations".  Each dict has a "from" field ("human" / "gpt") and a
    "value" field with the text.  We keep only the first human utterance —
    this is the inference prompt.

    The HuggingFace repo stores raw JSON files rather than a structured
    dataset, so we point the loader at the file explicitly.
    """
    print(f"Loading dataset '{dataset_name}' …")
    data_file = f"hf://datasets/{dataset_name}/ShareGPT_V3_unfiltered_cleaned_split.json"
    ds = load_dataset("json", data_files=data_file, split="train")

    prompts = []
    for row in ds:
        convs = row.get("conversations") or row.get("items") or []
        for turn in convs:
            speaker = (turn.get("from") or turn.get("role") or "").lower()
            if speaker in ("human", "user"):
                text = (turn.get("value") or turn.get("content") or "").strip()
                if text:
                    prompts.append(text)
                break   # first human turn only

    print(f"  Extracted {len(prompts):,} first-turn prompts.")
    return prompts


# ---------------------------------------------------------------------------
# 2. Tokenize prompts
# ---------------------------------------------------------------------------

def tokenize_prompts(
    prompts: List[str],
    tokenizer_name: str,
) -> List[Tuple[str, int]]:
    """
    Return (prompt_text, token_count) pairs using the Qwen3-32B tokenizer.

    add_special_tokens=False: lengths reflect the prompt body only.
    Special tokens (BOS/EOS) are added by the inference framework at run time.
    """
    print(f"\nLoading tokenizer '{tokenizer_name}' …")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    print(f"Tokenizing {len(prompts):,} prompts …")
    encodings = tokenizer(
        prompts,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )
    lengths = [len(ids) for ids in encodings["input_ids"]]
    print(f"  Done.  Length range: {min(lengths)}–{max(lengths)} tokens.")
    return list(zip(prompts, lengths))


# ---------------------------------------------------------------------------
# 3. Length distribution analysis
# ---------------------------------------------------------------------------

def analyse_lengths(
    prompt_lengths: List[Tuple[str, int]],
    plots_dir: str,
) -> None:
    lengths = [l for _, l in prompt_lengths]
    arr = np.array(lengths)

    print("\n=== Length distribution (unfiltered) ===")
    print(f"  Count  : {len(arr):,}")
    print(f"  Mean   : {arr.mean():.1f}")
    print(f"  Median : {np.median(arr):.1f}")
    print(f"  P90    : {np.percentile(arr, 90):.1f}")
    print(f"  P95    : {np.percentile(arr, 95):.1f}")
    print(f"  Max    : {arr.max()}")

    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(np.clip(arr, 0, 2000), bins=80, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Token length (clipped at 2 000)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("ShareGPT prompt length distribution (unfiltered)")

    filtered = arr[(arr >= MIN_TOKENS) & (arr <= MAX_TOKENS)]
    axes[1].hist(filtered, bins=60, color="darkorange", edgecolor="white")
    axes[1].set_xlabel("Token length")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"After filtering [{MIN_TOKENS}, {MAX_TOKENS}] tokens "
                      f"(n={len(filtered):,})")

    plt.tight_layout()
    path = os.path.join(plots_dir, "length_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Length distribution plot saved → {path}")


# ---------------------------------------------------------------------------
# 4. Filter prompts
# ---------------------------------------------------------------------------

def filter_prompts(
    prompt_lengths: List[Tuple[str, int]],
    min_tokens: int,
    max_tokens: int,
) -> List[Tuple[str, int]]:
    filtered = [(p, l) for p, l in prompt_lengths if min_tokens <= l <= max_tokens]
    print(f"\nFiltering [{min_tokens}, {max_tokens}] tokens: "
          f"{len(prompt_lengths):,} → {len(filtered):,} prompts retained.")
    return filtered


# ---------------------------------------------------------------------------
# 5. Batching with joint pad% + CoV control
# ---------------------------------------------------------------------------

def compute_padding_ratio(lengths: List[int]) -> float:
    """
    pad_ratio = (bs × max_len − Σ lengths) / (bs × max_len)

    0.0 → no wasted tokens (all sequences same length as max).
    0.70 → 70% of token slots are padding.
    """
    if not lengths:
        return 0.0
    max_len = max(lengths)
    return (len(lengths) * max_len - sum(lengths)) / (len(lengths) * max_len)


def compute_cov(lengths: List[int]) -> float:
    """Coefficient of variation: std / mean of sequence lengths in a batch."""
    if len(lengths) < 2:
        return 0.0
    arr = np.array(lengths, dtype=float)
    mean = arr.mean()
    return float(arr.std() / mean) if mean > 0 else 0.0


def find_closest_available(
    sorted_lengths: np.ndarray,
    target: float,
    available: np.ndarray,
) -> int:
    """
    Return the index (into sorted_lengths / sorted_pool) of the available
    sequence whose length is closest to `target`.

    Uses binary search to find the insertion point, then expands outward
    in both directions, always picking the closer of the two candidates.
    Returns -1 if no available sequence exists (should not happen in practice).
    """
    n = len(sorted_lengths)
    pos = int(np.searchsorted(sorted_lengths, target))
    lo, hi = pos - 1, pos

    while lo >= 0 or hi < n:
        d_lo = abs(sorted_lengths[lo] - target) if lo >= 0 else float("inf")
        d_hi = abs(sorted_lengths[hi] - target) if hi < n else float("inf")

        if d_lo <= d_hi:
            if lo >= 0 and available[lo]:
                return lo
            lo -= 1
        else:
            if hi < n and available[hi]:
                return hi
            hi += 1

    return -1


def make_batches_joint(
    pool: List[Tuple[str, int]],
    batch_size: int,
    n_batches: int,
    target_pad_ratio: float,
    target_cov: float,
    rng: np.random.Generator,
) -> List[List[Tuple[str, int]]]:
    """
    Build batches that jointly target a padding ratio AND an intra-batch CoV.

    The two dimensions are controlled independently:

    Padding ratio → fixes the relationship between anchor length and mean
    fill length.  Derived from the padding formula:

        pad_ratio = (bs × L_anchor − L_anchor − (bs−1) × mean_fill) / (bs × L_anchor)

    Solving for mean_fill:

        mean_fill = L_anchor × fill_factor
        fill_factor = (bs × (1 − pad_ratio) − 1) / (bs − 1)

    CoV → fixes the spread of fill lengths around their mean:

        fill_std = target_cov × fill_mean

    For each batch:
      1. Select one anchor sequence (the longest in its batch; sets max_len).
         The ideal anchor length is chosen so that fill_mean lands near the
         pool median — the densest region of the distribution, ensuring there
         are always plenty of fill candidates nearby.
      2. Sample (batch_size − 1) target fill lengths from:
            N(fill_mean, fill_std²)  clamped to [MIN_TOKENS, L_anchor]
         Gaussian sampling decouples the fill selection from ShareGPT's skew:
         whatever the distribution shape, the mean and std of the selected
         fills will converge to (fill_mean, fill_std).
      3. For each sampled target length, pick the closest sequence in the
         pool (without replacement within a batch; WITH replacement across
         batches, so the same prompt can appear in multiple batches).

    Reuse policy
    ------------
    Fills are drawn without replacement within a single batch (each sequence
    appears at most once per batch), but the pool is reset between batches.
    This avoids the "pool exhaustion" problem that caused high variance in the
    previous version, and is valid because the batching strategy — not prompt
    content — is the experimental variable.
    """
    # Sort pool by length for efficient binary-search-based nearest-neighbour.
    sort_order    = np.argsort([item[1] for item in pool])
    sorted_pool   = [pool[i] for i in sort_order]
    sorted_lengths = np.array([item[1] for item in sorted_pool], dtype=float)

    # fill_factor: fraction of L_anchor that gives the desired mean_fill.
    fill_factor = (batch_size * (1.0 - target_pad_ratio) - 1.0) / (batch_size - 1)
    fill_factor = max(fill_factor, 0.01)   # guard: don't allow near-zero fill mean

    # Ideal anchor length: the value of L such that fill_factor × L equals
    # the pool median.  This ensures fill targets land in the densest part of
    # the distribution for any pad level.
    pool_median   = float(np.median(sorted_lengths))
    ideal_anchor  = np.clip(pool_median / fill_factor,
                            sorted_lengths[0], sorted_lengths[-1])

    # Select n_batches anchors: sequences closest to ideal_anchor, no replacement.
    anchor_dists  = np.abs(sorted_lengths - ideal_anchor)
    anchor_positions = np.argsort(anchor_dists)[:n_batches]   # indices in sorted array

    batches = []
    for anchor_pos in anchor_positions:
        L_anchor    = float(sorted_lengths[anchor_pos])
        anchor_item = sorted_pool[anchor_pos]

        fill_mean = L_anchor * fill_factor
        fill_std  = target_cov * fill_mean

        # Sample target fill lengths from N(fill_mean, fill_std²).
        # Clamp to [MIN_TOKENS, L_anchor] so no fill exceeds the anchor
        # (which would change which sequence is actually the max in the batch).
        n_fills = batch_size - 1
        raw     = rng.normal(fill_mean, fill_std, size=n_fills * 4)
        clamped = np.clip(raw, MIN_TOKENS, L_anchor)
        rng.shuffle(clamped)
        target_lens = clamped[:n_fills]

        # Reset availability for this batch (fills are reused across batches).
        available = np.ones(len(sorted_pool), dtype=bool)
        available[anchor_pos] = False   # anchor cannot also be a fill

        fills = []
        for tlen in target_lens:
            idx = find_closest_available(sorted_lengths, tlen, available)
            if idx < 0:
                break
            fills.append(sorted_pool[idx])
            available[idx] = False

        if len(fills) < n_fills:
            continue   # couldn't fill batch; skip (rare edge case)

        batches.append(fills + [anchor_item])

    return batches


# ---------------------------------------------------------------------------
# 6. Build 4×4 benchmark dataset
# ---------------------------------------------------------------------------

def build_benchmark_dataset(
    pool: List[Tuple[str, int]],
    pad_levels: List[float],
    cov_levels: List[float],
    batch_size: int,
    n_batches_per_cell: int,
    output_dir: str,
    seed: int,
) -> pd.DataFrame:
    """
    Construct the full 4×4 benchmark grid.

    For every (pad_level, cov_level) cell, build n_batches_per_cell batches
    and save them as a JSON file.  All 16 cells draw from the same prompt
    pool — the pool content is held constant; only the batching differs.

    JSON format per cell:
    [
      {
        "batch_id": 0,
        "target_pad_ratio": 0.10,
        "target_cov": 0.05,
        "actual_pad_ratio": 0.097,
        "actual_cov": 0.048,
        "actual_mean_len": 87.3,
        "actual_std_len": 4.2,
        "prompts": [
          {"prompt": "...", "token_length": 85},
          ...   // batch_size entries
        ]
      },
      ...   // n_batches_per_cell entries
    ]
    """
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []

    for pad in pad_levels:
        for cov in cov_levels:
            pad_tag   = f"pad{int(pad * 100):02d}"
            cov_tag   = f"cov{int(cov * 100):02d}"
            cell_name = f"{pad_tag}_{cov_tag}"

            print(f"\n--- Cell {cell_name}  (pad={pad:.0%}, CoV={cov:.2f}) ---")

            batches = make_batches_joint(
                pool, batch_size, n_batches_per_cell, pad, cov, rng
            )

            cell_data = []
            for batch_id, batch in enumerate(batches):
                lengths      = [l for _, l in batch]
                fill_lengths = lengths[:-1]   # last item is the anchor

                actual_pad = compute_padding_ratio(lengths)
                actual_cov = compute_cov(fill_lengths)
                mean_len   = float(np.mean(lengths))
                std_len    = float(np.std(lengths))

                cell_data.append({
                    "batch_id":         batch_id,
                    "target_pad_ratio": pad,
                    "target_cov":       cov,
                    "actual_pad_ratio": round(actual_pad, 4),
                    "actual_cov":       round(actual_cov, 4),
                    "actual_mean_len":  round(mean_len, 2),
                    "actual_std_len":   round(std_len, 2),
                    "prompts": [
                        {"prompt": text, "token_length": length}
                        for text, length in batch
                    ],
                })

                summary_rows.append({
                    "cell":             cell_name,
                    "pad_level":        pad,
                    "cov_level":        cov,
                    "batch_id":         batch_id,
                    "actual_pad_ratio": round(actual_pad, 4),
                    "actual_cov":       round(actual_cov, 4),
                    "actual_mean_len":  round(mean_len, 2),
                    "actual_std_len":   round(std_len, 2),
                })

                print(f"  batch {batch_id}: pad={actual_pad:.3f} "
                      f"(Δ={actual_pad - pad:+.3f}), "
                      f"CoV={actual_cov:.3f} (Δ={actual_cov - cov:+.3f})")

            out_path = os.path.join(output_dir, f"{cell_name}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(cell_data, f, ensure_ascii=False, indent=2)
            print(f"  Saved → {out_path}")

    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(output_dir, "benchmark_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary CSV saved → {csv_path}")
    return df


# ---------------------------------------------------------------------------
# 7. Plots and summary table
# ---------------------------------------------------------------------------

def plot_benchmark_grid(
    df: pd.DataFrame,
    pad_levels: List[float],
    cov_levels: List[float],
    plots_dir: str,
) -> None:
    """
    Two side-by-side 4×4 heatmaps:
      Left  — mean achieved padding ratio per cell (target: x-axis)
      Right — mean achieved CoV per cell (target: y-axis)

    Cell annotations show both the achieved value and the signed error
    relative to the target, so it is immediately obvious which cells
    hit their targets and which drift.
    """
    os.makedirs(plots_dir, exist_ok=True)

    cell_means = df.groupby(["pad_level", "cov_level"])[
        ["actual_pad_ratio", "actual_cov"]
    ].mean()

    # Build 2-D grids  (rows = CoV levels, columns = pad levels)
    pad_grid = np.full((len(cov_levels), len(pad_levels)), np.nan)
    cov_grid = np.full((len(cov_levels), len(pad_levels)), np.nan)

    for ci, pad in enumerate(pad_levels):
        for ri, cov in enumerate(cov_levels):
            try:
                row = cell_means.loc[(pad, cov)]
                pad_grid[ri, ci] = row["actual_pad_ratio"]
                cov_grid[ri, ci] = row["actual_cov"]
            except KeyError:
                pass

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    configs = [
        (axes[0], pad_grid, "Achieved padding ratio",
         np.array([[p] * len(cov_levels) for p in pad_levels]).T, "RdYlGn_r", 0.0, 1.0),
        (axes[1], cov_grid, "Achieved CoV",
         np.array([[c] * len(pad_levels) for c in cov_levels]),   "RdYlGn_r", 0.0, 1.0),
    ]

    pad_labels = [f"{int(p * 100)}%" for p in pad_levels]
    cov_labels = [f"{c:.2f}" for c in cov_levels]

    for ax, grid, title, target_grid, cmap, vmin, vmax in configs:
        im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pad_levels)))
        ax.set_xticklabels(pad_labels, fontsize=10)
        ax.set_yticks(range(len(cov_levels)))
        ax.set_yticklabels(cov_labels, fontsize=10)
        ax.set_xlabel("Target pad%", fontsize=11)
        ax.set_ylabel("Target CoV", fontsize=11)
        ax.set_title(title, fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate: "achieved\n(Δerror)"
        for ri in range(len(cov_levels)):
            for ci in range(len(pad_levels)):
                achieved = grid[ri, ci]
                target   = target_grid[ri, ci]
                if not np.isnan(achieved):
                    ax.text(ci, ri,
                            f"{achieved:.2f}\n({achieved - target:+.2f})",
                            ha="center", va="center", fontsize=7.5,
                            color="black")

    plt.suptitle("Achieved vs target values across the 4×4 benchmark grid",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(plots_dir, "benchmark_grid.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Benchmark grid plot saved → {path}")


def print_summary_table(
    df: pd.DataFrame,
    pad_levels: List[float],
    cov_levels: List[float],
) -> None:
    print("\n" + "=" * 76)
    print("SUMMARY  —  mean achieved values per cell  [rows=CoV target, cols=pad target]")
    print("=" * 76)

    pivot_pad = df.pivot_table(
        values="actual_pad_ratio",
        index="cov_level",
        columns="pad_level",
        aggfunc="mean",
    ).reindex(index=cov_levels, columns=pad_levels)

    pivot_cov = df.pivot_table(
        values="actual_cov",
        index="cov_level",
        columns="pad_level",
        aggfunc="mean",
    ).reindex(index=cov_levels, columns=pad_levels)

    pivot_pad.columns = [f"pad={int(p*100)}%" for p in pivot_pad.columns]
    pivot_pad.index   = [f"CoV={c:.2f}" for c in pivot_pad.index]
    pivot_cov.columns = [f"pad={int(p*100)}%" for p in pivot_cov.columns]
    pivot_cov.index   = [f"CoV={c:.2f}" for c in pivot_cov.index]

    print("\nMean achieved padding ratio (target col headers):")
    print(pivot_pad.to_string(float_format="{:.3f}".format))

    print("\nMean achieved CoV (target row index):")
    print(pivot_cov.to_string(float_format="{:.3f}".format))
    print("=" * 76)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    # 1. Load
    prompts = load_sharegpt_prompts(DATASET_NAME)

    # 2. Tokenize
    prompt_lengths = tokenize_prompts(prompts, TOKENIZER_NAME)

    # 3. Analyse
    analyse_lengths(prompt_lengths, PLOTS_DIR)

    # 4. Filter
    filtered = filter_prompts(prompt_lengths, MIN_TOKENS, MAX_TOKENS)

    # 5–6. Build 4×4 grid and save
    df = build_benchmark_dataset(
        pool               = filtered,
        pad_levels         = PAD_LEVELS,
        cov_levels         = COV_LEVELS,
        batch_size         = BATCH_SIZE,
        n_batches_per_cell = N_BATCHES_PER_CELL,
        output_dir         = OUTPUT_DIR,
        seed               = SEED,
    )

    # 7. Plots and table
    plot_benchmark_grid(df, PAD_LEVELS, COV_LEVELS, PLOTS_DIR)
    print_summary_table(df, PAD_LEVELS, COV_LEVELS)

    print(f"\nAll outputs written to ./{OUTPUT_DIR}/  and  ./{PLOTS_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
