"""
preprocess_dataset.py

Prepares the ShareGPT dataset for controlled padding ratio experiments.

Pipeline:
  1. Load ShareGPT from HuggingFace, extract first conversational turn.
  2. Tokenize every prompt with the Qwen3-32B tokenizer.
  3. Analyse and plot the length distribution.
  4. Filter to [32, 1024] tokens.
  5. Build four prompt groups (targeting ~0%, 25%, 50%, 75% padding) by
     batching the *same* pool of prompts with different grouping strategies.
  6. Save per-group JSON files and a summary CSV.
  7. Print a summary table and produce plots.

Key design note
---------------
The prompts themselves are NOT modified.  Padding arises solely from grouping
sequences of unequal length into fixed-size batches and padding each batch to
its longest sequence.  The four groups use the same underlying prompt pool —
only the batching strategy differs, which is the experimental manipulation.

To run: 
#Dependencies: 
    pip install transformers datasets matplotlib numpy pandas
#Script:
    python preprocess_dataset.py

Output structure:
benchmark_dataset/
  group_low.json          # ~128 records, near-zero padding
  group_low_mid.json      # ~128 records, ~25% padding
  group_mid.json          # ~128 records, ~50% padding
  group_high.json         # ~128 records, ~75% padding
  benchmark_summary.csv

plots/
  length_distribution.png
  padding_ratio_by_group.png
"""

import json
import os
import random
from collections import defaultdict
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")          # headless — safe on GPU servers with no display
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

BATCH_SIZE  = 16
BATCHES_PER_GROUP = 8               # 8 × 16 = 128 prompts per group; ≈500 total
PROMPTS_PER_GROUP = BATCH_SIZE * BATCHES_PER_GROUP   # 128

TARGET_RATIOS = {
    "low":    0.00,   # bucket-batching → near-zero padding in practice
    "low_mid": 0.25,
    "mid":    0.50,
    "high":   0.75,
}

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
    "conversations" (or "items" depending on the snapshot).  Each dict has
    a "from" field ("human" / "gpt") and a "value" field with the text.
    We keep only the very first human utterance — this is our inference prompt.
    """
    print(f"Loading dataset '{dataset_name}' …")
    ds = load_dataset(dataset_name, split="train")

    prompts = []
    for row in ds:
        convs = row.get("conversations") or row.get("items") or []
        for turn in convs:
            speaker = (turn.get("from") or turn.get("role") or "").lower()
            if speaker in ("human", "user"):
                text = (turn.get("value") or turn.get("content") or "").strip()
                if text:
                    prompts.append(text)
                break   # only the first human turn per conversation

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
    Return (prompt_text, token_count) pairs using the specified tokenizer.

    We use `add_special_tokens=False` so that the length reflects only the
    prompt body — special tokens (BOS/EOS) are typically added by the
    inference framework at run time.
    """
    print(f"\nLoading tokenizer '{tokenizer_name}' …")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    print(f"Tokenizing {len(prompts):,} prompts (batch mode) …")
    # Batch encoding without padding/truncation — we want true lengths.
    encodings = tokenizer(
        prompts,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )
    lengths = [len(ids) for ids in encodings["input_ids"]]

    result = list(zip(prompts, lengths))
    print(f"  Done.  Length range: {min(lengths)}–{max(lengths)} tokens.")
    return result


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

    # Full distribution (clipped to 2 000 for readability)
    axes[0].hist(np.clip(arr, 0, 2000), bins=80, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Token length (clipped at 2 000)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("ShareGPT prompt length distribution (unfiltered)")

    # Filtered range
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
# 5. Batching strategies
# ---------------------------------------------------------------------------

def compute_padding_ratio(lengths: List[int]) -> float:
    """
    Padding ratio for a single batch:

        (batch_size × max_len − Σ actual_lengths) / (batch_size × max_len)

    A ratio of 0.0 means every sequence is as long as the longest one
    (zero wasted tokens).  A ratio of 0.75 means 75 % of token slots are
    padding.
    """
    if not lengths:
        return 0.0
    max_len = max(lengths)
    total_slots = len(lengths) * max_len
    padding_tokens = total_slots - sum(lengths)
    return padding_tokens / total_slots


def make_batches_low_padding(
    pool: List[Tuple[str, int]],
    batch_size: int,
    n_batches: int,
) -> List[List[Tuple[str, int]]]:
    """
    Low-padding strategy (bucket / sort batching).

    Sort all prompts by token length and slice contiguous windows.
    Sequences that land in the same batch have very similar lengths, so
    the longest sequence in the batch is only marginally longer than the
    others — minimising wasted padding tokens.
    """
    sorted_pool = sorted(pool, key=lambda x: x[1])
    batches = []
    for i in range(n_batches):
        start = i * batch_size
        batch = sorted_pool[start : start + batch_size]
        if len(batch) == batch_size:
            batches.append(batch)
    return batches


def make_batches_high_padding(
    pool: List[Tuple[str, int]],
    batch_size: int,
    n_batches: int,
) -> List[List[Tuple[str, int]]]:
    """
    High-padding strategy (shortest + longest interleaving).

    Sort by length, then pair the shortest half with the longest half so
    that each batch contains the maximum possible length variance.  Within
    each batch the max_len is dictated by a very long sequence while most
    other sequences are much shorter — creating a large padding ratio.

    Concretely: given sorted indices [0 … N-1], batch k is formed by
    taking stride-spaced elements from the bottom and top of the sorted
    list simultaneously.
    """
    sorted_pool = sorted(pool, key=lambda x: x[1])
    n = len(sorted_pool)
    half = n // 2

    short_half = sorted_pool[:half]
    long_half  = sorted_pool[half:]

    # Take (batch_size // 2) from the short end and (batch_size // 2) from
    # the long end for each batch.  This pairs short and long sequences.
    shorts_per_batch = batch_size // 2
    longs_per_batch  = batch_size - shorts_per_batch

    batches = []
    for i in range(n_batches):
        s_start = i * shorts_per_batch
        l_start = i * longs_per_batch
        short_slice = short_half[s_start : s_start + shorts_per_batch]
        long_slice  = long_half [l_start : l_start + longs_per_batch]
        batch = short_slice + long_slice
        if len(batch) == batch_size:
            batches.append(batch)
    return batches


def make_batches_mid_padding(
    pool: List[Tuple[str, int]],
    batch_size: int,
    n_batches: int,
) -> List[List[Tuple[str, int]]]:
    """
    Mid-padding strategy (interleaved stride batching).

    Sort by length and then use a stride of `n_batches` when assigning
    prompts to batches.  Each batch therefore contains sequences spread
    evenly across the sorted order — e.g. one very short, one short-mid,
    one mid-long, one long — producing moderate within-batch length
    variance and intermediate padding ratios.

    The stride width controls how spread out the lengths are:
      - stride = 1  →  contiguous windows  (low padding, like bucket)
      - stride = n_batches  →  maximally spread  (higher padding)
    We use stride = n_batches as the mid-point between the two extremes.
    """
    sorted_pool = sorted(pool, key=lambda x: x[1])
    batches: List[List] = [[] for _ in range(n_batches)]
    for idx, item in enumerate(sorted_pool[: n_batches * batch_size]):
        # Round-robin assignment across batches based on sorted position
        batches[idx % n_batches].append(item)
    return [b for b in batches if len(b) == batch_size][:n_batches]


# ---------------------------------------------------------------------------
# Dispatcher: choose strategy by target ratio
# ---------------------------------------------------------------------------

STRATEGY_THRESHOLDS = {
    # target_ratio → batching function
    0.00: make_batches_low_padding,
    0.25: make_batches_mid_padding,
    0.50: make_batches_high_padding,
    0.75: make_batches_high_padding,
}

def build_batches_for_group(
    pool: List[Tuple[str, int]],
    target_ratio: float,
    batch_size: int,
    n_batches: int,
) -> List[List[Tuple[str, int]]]:
    """Select and run the batching strategy closest to the target ratio."""
    # Pick the strategy whose key is nearest to the target ratio.
    key = min(STRATEGY_THRESHOLDS.keys(), key=lambda k: abs(k - target_ratio))
    strategy_fn = STRATEGY_THRESHOLDS[key]
    return strategy_fn(pool, batch_size, n_batches)


# ---------------------------------------------------------------------------
# 6. Construct benchmark dataset
# ---------------------------------------------------------------------------

def build_benchmark_dataset(
    prompt_lengths: List[Tuple[str, int]],
    target_ratios: dict,
    batch_size: int,
    batches_per_group: int,
    output_dir: str,
    plots_dir: str,
    seed: int,
) -> pd.DataFrame:
    """
    Build the full benchmark dataset.

    All groups share the *same* pool of prompts — only the batching differs.
    This is the key experimental control: any throughput differences between
    groups cannot be attributed to prompt content, only to padding structure.
    """
    rng = random.Random(seed)

    # How many prompts do we need in total for one group?
    prompts_needed = batch_size * batches_per_group

    # Shuffle deterministically, then take the first `prompts_needed` entries
    # as the shared pool.  All four groups will use this same pool.
    shuffled = list(prompt_lengths)
    rng.shuffle(shuffled)
    shared_pool = shuffled[:prompts_needed]

    print(f"\nShared pool size: {len(shared_pool)} prompts "
          f"({batches_per_group} batches × {batch_size} prompts/batch)")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    records = []           # for the summary CSV
    group_achieved = {}    # group_name → list of achieved padding ratios

    for group_name, target in target_ratios.items():
        print(f"\n--- Group '{group_name}' (target={target:.0%}) ---")

        batches = build_batches_for_group(
            pool=shared_pool,
            target_ratio=target,
            batch_size=batch_size,
            n_batches=batches_per_group,
        )

        group_records = []
        achieved_ratios = []
        prompt_id_counter = 0

        for batch_id, batch in enumerate(batches):
            lengths_in_batch = [l for _, l in batch]
            achieved_ratio   = compute_padding_ratio(lengths_in_batch)
            achieved_ratios.append(achieved_ratio)

            for prompt_text, token_len in batch:
                group_records.append({
                    "prompt":               prompt_text,
                    "token_length":         token_len,
                    "batch_id":             batch_id,
                    "actual_padding_ratio": round(achieved_ratio, 4),
                })
                records.append({
                    "prompt_id":            prompt_id_counter,
                    "token_length":         token_len,
                    "padding_group":        group_name,
                    "batch_id":             batch_id,
                    "actual_padding_ratio": round(achieved_ratio, 4),
                })
                prompt_id_counter += 1

        group_achieved[group_name] = achieved_ratios

        mean_r = np.mean(achieved_ratios)
        std_r  = np.std(achieved_ratios)
        print(f"  Batches formed : {len(batches)}")
        print(f"  Achieved ratio : {mean_r:.3f} ± {std_r:.3f}  "
              f"(target was {target:.2f})")

        out_path = os.path.join(output_dir, f"group_{group_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(group_records, f, ensure_ascii=False, indent=2)
        print(f"  Saved → {out_path}")

    # ------------------------------------------------------------------
    # Summary CSV
    # ------------------------------------------------------------------
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "benchmark_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary CSV saved → {csv_path}")

    # ------------------------------------------------------------------
    # Plot: achieved vs target padding ratio per group
    # ------------------------------------------------------------------
    plot_padding_ratios(group_achieved, target_ratios, plots_dir)

    return df


# ---------------------------------------------------------------------------
# 7. Plots and summary table
# ---------------------------------------------------------------------------

def plot_padding_ratios(
    group_achieved: dict,
    target_ratios: dict,
    plots_dir: str,
) -> None:
    """Box-plot of achieved padding ratios with target lines overlaid."""
    fig, ax = plt.subplots(figsize=(10, 6))

    group_names = list(group_achieved.keys())
    data = [group_achieved[g] for g in group_names]
    targets = [target_ratios[g] for g in group_names]

    bp = ax.boxplot(data, patch_artist=True, notch=False)
    colours = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)

    # Overlay target lines
    for i, target in enumerate(targets, start=1):
        ax.hlines(target, i - 0.4, i + 0.4,
                  colors="black", linestyles="--", linewidths=1.5,
                  label="Target" if i == 1 else "")

    ax.set_xticks(range(1, len(group_names) + 1))
    ax.set_xticklabels(
        [f"{g}\n(target={target_ratios[g]:.0%})" for g in group_names],
        fontsize=10,
    )
    ax.set_ylabel("Achieved padding ratio")
    ax.set_title("Achieved vs target padding ratio per benchmark group")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    path = os.path.join(plots_dir, "padding_ratio_by_group.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Padding-ratio plot saved → {path}")


def print_summary_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY: achieved padding ratio per group")
    print("=" * 60)

    table = (
        df.groupby("padding_group")["actual_padding_ratio"]
        .agg(["mean", "std", "min", "max", "count"])
        .rename(columns={
            "mean":  "Mean",
            "std":   "Std",
            "min":   "Min",
            "max":   "Max",
            "count": "N prompts",
        })
    )
    # Reorder rows to match TARGET_RATIOS insertion order
    order = list(TARGET_RATIOS.keys())
    table = table.reindex([g for g in order if g in table.index])

    fmt_cols = ["Mean", "Std", "Min", "Max"]
    for col in fmt_cols:
        table[col] = table[col].map("{:.3f}".format)

    print(table.to_string())
    print("=" * 60)


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

    # 3. Analyse (plots unfiltered + filtered length distribution)
    analyse_lengths(prompt_lengths, PLOTS_DIR)

    # 4. Filter
    filtered = filter_prompts(prompt_lengths, MIN_TOKENS, MAX_TOKENS)

    if len(filtered) < PROMPTS_PER_GROUP:
        raise ValueError(
            f"Not enough prompts after filtering: need {PROMPTS_PER_GROUP}, "
            f"got {len(filtered)}.  Relax MIN/MAX_TOKENS or reduce BATCHES_PER_GROUP."
        )

    # 5–6. Build groups and save
    df = build_benchmark_dataset(
        prompt_lengths   = filtered,
        target_ratios    = TARGET_RATIOS,
        batch_size       = BATCH_SIZE,
        batches_per_group= BATCHES_PER_GROUP,
        output_dir       = OUTPUT_DIR,
        plots_dir        = PLOTS_DIR,
        seed             = SEED,
    )

    # 7. Summary table
    print_summary_table(df)

    print(f"\nAll outputs written to ./{OUTPUT_DIR}/  and  ./{PLOTS_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
