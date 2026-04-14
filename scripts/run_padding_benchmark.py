"""
run_padding_benchmark.py

Padding-effects inference benchmark for the CMSC 828G project.

Loads pre-built benchmark cells from benchmark_dataset/ (produced by
preprocess_dataset.py), feeds each batch through a Megatron-LM static
inference engine, and records timing + throughput metrics.

Key measurements per batch
--------------------------
  wall_clock_s         : seconds for one generate() call (prefill + 1 decode)
  total_tokens         : batch_size × max_length (all slots including padding)
  real_tokens          : sum of actual prompt lengths (non-padding)
  pad_tokens           : total_tokens − real_tokens
  raw_tps              : total_tokens / wall_clock_s
  effective_tps        : real_tokens / wall_clock_s
  max_memory_gb        : peak GPU memory on this rank after the batch

Usage (run from Megatron-LM repo root so gpt_builders / model_provider are importable):

    torchrun --nproc_per_node=4 scripts/run_padding_benchmark.py \\
        --tensor-model-parallel-size 4 \\
        --pipeline-model-parallel-size 1 \\
        ... <Qwen3-32B arch flags> ... \\
        --load /path/to/qwen3-32b-mcore-tp4 \\
        --tokenizer-type HuggingFaceTokenizer \\
        --tokenizer-model /path/to/Qwen3-32B \\
        --benchmark-dataset-dir /path/to/benchmark_dataset \\
        --results-dir /path/to/results \\
        --config-name config_a \\
        --n-warmup 2

See launch_config_*.sh for full flag sets.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch

# ---------------------------------------------------------------------------
# Megatron repo root must be on sys.path so that gpt_builders / model_provider
# are importable.  This file lives at scripts/run_padding_benchmark.py, so the
# repo root is one level up from scripts/.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent / "Megatron-LM"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.engines import StaticInferenceEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.inference.utils import add_inference_args, get_model_for_inference
from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.initialize import initialize_megatron


# ---------------------------------------------------------------------------
# Extra CLI args
# ---------------------------------------------------------------------------

def add_benchmark_args(parser):
    add_inference_args(parser)

    grp = parser.add_argument_group("Padding benchmark")
    grp.add_argument(
        "--benchmark-dataset-dir",
        type=str,
        default="benchmark_dataset",
        help="Directory produced by preprocess_dataset.py (contains pad*_cov*.json files).",
    )
    grp.add_argument(
        "--benchmark-cell",
        type=str,
        default=None,
        help="Run only this cell (e.g. 'pad20_cov05').  Defaults to all 16 cells.",
    )
    grp.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory where per-cell results JSON and summary CSV are written.",
    )
    grp.add_argument(
        "--config-name",
        type=str,
        default="config",
        help="Label for this TP/PP configuration (e.g. 'config_a').  Used in output filenames.",
    )
    grp.add_argument(
        "--n-warmup",
        type=int,
        default=2,
        help="Number of warmup batches before timing starts (discarded).",
    )
    return parser


# ---------------------------------------------------------------------------
# Inference engine builder
# ---------------------------------------------------------------------------

def build_engine(args, model) -> StaticInferenceEngine:
    tokenizer = build_tokenizer(args)
    context = StaticInferenceContext(
        args.inference_max_requests,
        args.inference_max_seq_length,
    )
    wrapped = GPTInferenceWrapper(model, context)
    controller = TextGenerationController(
        inference_wrapped_model=wrapped, tokenizer=tokenizer
    )
    return StaticInferenceEngine(text_generation_controller=controller)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def load_cell(cell_path: str):
    """Return list of batch dicts from a cell JSON file."""
    with open(cell_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_batch_stats(batch_dict: dict):
    """
    Extract prompt texts and compute token counts from one batch dict.

    Returns
    -------
    prompts      : list[str]
    real_tokens  : int   (sum of actual prompt lengths)
    total_tokens : int   (batch_size × max_length, i.e., all padded slots)
    pad_tokens   : int
    """
    prompts_info = batch_dict["prompts"]
    lengths = [p["token_length"] for p in prompts_info]
    prompts = [p["prompt"] for p in prompts_info]

    real_tokens = sum(lengths)
    max_len = max(lengths)
    total_tokens = len(lengths) * max_len
    pad_tokens = total_tokens - real_tokens

    return prompts, real_tokens, total_tokens, pad_tokens


def run_batch(engine: StaticInferenceEngine, prompts: List[str]) -> float:
    """
    Run one generate() call and return wall-clock seconds.

    We request exactly 1 output token so the measurement is dominated by the
    prefill forward pass (which is where padding overhead lives).
    """
    sampling_params = SamplingParams(
        temperature=1.0,
        top_k=1,
        top_p=0.0,
        num_tokens_to_generate=1,
    )

    # Synchronize before timing to exclude any queued GPU work.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    engine.generate(prompts=prompts, sampling_params=sampling_params)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return t1 - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.inference_mode()
def main():
    initialize_megatron(
        extra_args_provider=add_benchmark_args,
        args_defaults={
            "no_load_rng": True,
            "no_load_optim": True,
            "micro_batch_size": 1,
            "exit_on_missing_checkpoint": True,
        },
    )

    args = get_args()
    is_rank0 = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    # Build model + engine.
    model = get_model_for_inference()
    engine = build_engine(args, model)

    dataset_dir = Path(args.benchmark_dataset_dir)
    results_dir = Path(args.results_dir)

    if is_rank0:
        results_dir.mkdir(parents=True, exist_ok=True)

    # Discover cell files.
    if args.benchmark_cell:
        cell_files = [dataset_dir / f"{args.benchmark_cell}.json"]
    else:
        cell_files = sorted(dataset_dir.glob("pad*.json"))

    if not cell_files:
        print_rank_0(f"No cell files found in {dataset_dir}")
        return

    all_results = []

    for cell_path in cell_files:
        cell_name = cell_path.stem  # e.g. "pad20_cov05"
        print_rank_0(f"\n{'=' * 60}")
        print_rank_0(f"Cell: {cell_name}  ({cell_path})")

        batches = load_cell(str(cell_path))

        # Warmup using first N batches (results discarded).
        warmup_batches = batches[: args.n_warmup]
        for wi, wbatch in enumerate(warmup_batches):
            prompts, *_ = compute_batch_stats(wbatch)
            print_rank_0(f"  [warmup {wi + 1}/{len(warmup_batches)}]")
            run_batch(engine, prompts)

        # Reset peak memory counter before timed batches.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Timed batches.
        timed_batches = batches[args.n_warmup :]
        if not timed_batches:
            print_rank_0("  WARNING: no timed batches left (all used for warmup)")
            continue

        for batch_dict in timed_batches:
            prompts, real_tokens, total_tokens, pad_tokens = compute_batch_stats(batch_dict)

            wall_clock_s = run_batch(engine, prompts)

            max_mem_bytes = (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            )
            max_mem_gb = max_mem_bytes / 1e9
            torch.cuda.reset_peak_memory_stats()

            raw_tps = total_tokens / wall_clock_s
            effective_tps = real_tokens / wall_clock_s

            record = {
                "config": args.config_name,
                "cell": cell_name,
                "batch_id": batch_dict["batch_id"],
                "target_pad_ratio": batch_dict["target_pad_ratio"],
                "target_cov": batch_dict["target_cov"],
                "actual_pad_ratio": batch_dict["actual_pad_ratio"],
                "actual_cov": batch_dict["actual_cov"],
                "actual_mean_len": batch_dict["actual_mean_len"],
                "actual_std_len": batch_dict["actual_std_len"],
                "num_prompts": len(prompts),
                "real_tokens": real_tokens,
                "total_tokens": total_tokens,
                "pad_tokens": pad_tokens,
                "wall_clock_s": round(wall_clock_s, 6),
                "raw_tps": round(raw_tps, 2),
                "effective_tps": round(effective_tps, 2),
                "max_memory_gb": round(max_mem_gb, 3),
            }
            all_results.append(record)

            print_rank_0(
                f"  batch {batch_dict['batch_id']:02d} | "
                f"pad={batch_dict['actual_pad_ratio']:.3f} "
                f"cov={batch_dict['actual_cov']:.3f} | "
                f"real={real_tokens} total={total_tokens} | "
                f"time={wall_clock_s:.3f}s | "
                f"eff_tps={effective_tps:.1f} raw_tps={raw_tps:.1f} | "
                f"mem={max_mem_gb:.2f}GB"
            )

    # Write results (rank 0 only).
    if is_rank0 and all_results:
        out_json = results_dir / f"{args.config_name}_results.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print_rank_0(f"\nResults written → {out_json}")

        # Also write a flat CSV for easy analysis.
        import csv

        out_csv = results_dir / f"{args.config_name}_results.csv"
        fieldnames = list(all_results[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print_rank_0(f"Results written → {out_csv}")

    print_rank_0("\nDone.")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
