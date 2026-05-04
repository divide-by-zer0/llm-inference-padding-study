"""
run_padding_benchmark.py

Padding-effects inference benchmark for the CMSC 828G project.

Loads pre-built benchmark cells from benchmark_dataset/ (produced by
preprocess_dataset.py), feeds each batch through a Megatron-LM static
inference engine, and records timing + throughput metrics.

Key measurements per batch
--------------------------
  wall_clock_s         : seconds for one generate() call (prefill + decode)
  prompt_total_tokens  : batch_size x max_prompt_length (including padding)
  prompt_real_tokens   : sum of actual prompt lengths (non-padding)
  prompt_pad_tokens    : prompt_total_tokens - prompt_real_tokens
  generated_tokens     : batch_size x num_decode_tokens
  decode_tps           : generated_tokens / wall_clock_s
  end_to_end_tps       : (prompt_real_tokens + generated_tokens) / wall_clock_s
  max_memory_gb        : peak GPU memory on this rank after the batch

Usage (run from Megatron-LM repo root so gpt_builders / model_provider are importable):

    torchrun --nproc_per_node=4 scripts/run_padding_benchmark.py \\
        --tensor-model-parallel-size 4 \\
        --pipeline-model-parallel-size 1 \\
        ... <Qwen2.5-32B arch flags> ... \\
        --load /path/to/qwen2.5-32b-mcore-tp4 \\
        --tokenizer-type HuggingFaceTokenizer \\
        --tokenizer-model /path/to/Qwen2.5-32B \\
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
from megatron.training.arguments import parse_and_validate_args
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
    grp.add_argument(
        "--num-decode-tokens",
        type=int,
        default=512,
        help="Number of new tokens to generate per prompt for decode throughput.",
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


def _timed_generate(engine: StaticInferenceEngine, prompts: List[str], num_tokens: int) -> float:
    sampling_params = SamplingParams(
        temperature=1.0,
        top_k=1,
        top_p=0.0,
        num_tokens_to_generate=num_tokens,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    engine.generate(prompts=prompts, sampling_params=sampling_params)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def run_batch(
    engine: StaticInferenceEngine,
    prompts: List[str],
    num_decode_tokens: int,
) -> tuple:
    """
    Run two generate() calls and return (ttft_s, wall_clock_s).

    ttft_s       : prefill + 1 decode step (proxy for time-to-first-token)
    wall_clock_s : prefill + num_decode_tokens decode steps (total time)
    generation_time_s is derived as wall_clock_s - ttft_s by the caller.
    """
    ttft_s = _timed_generate(engine, prompts, 1)
    wall_clock_s = _timed_generate(engine, prompts, num_decode_tokens)
    return ttft_s, wall_clock_s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.inference_mode()
def main():
    args = parse_and_validate_args(
        extra_args_provider=add_benchmark_args,
        args_defaults={
            "no_load_rng": True,
            "no_load_optim": True,
            "micro_batch_size": 1,
            "exit_on_missing_checkpoint": True,
        },
    )

    initialize_megatron()
    is_rank0 = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    if args.num_decode_tokens <= 0:
        raise ValueError("--num-decode-tokens must be positive")

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
            run_batch(engine, prompts, args.num_decode_tokens)  # results discarded

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

            ttft_s, wall_clock_s = run_batch(engine, prompts, args.num_decode_tokens)
            generation_time_s = wall_clock_s - ttft_s

            max_mem_bytes = (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            )
            max_mem_gb = max_mem_bytes / 1e9
            torch.cuda.reset_peak_memory_stats()

            generated_tokens = len(prompts) * args.num_decode_tokens
            decode_tokens = len(prompts) * (args.num_decode_tokens - 1)
            decode_tps = decode_tokens / generation_time_s
            end_to_end_tps = (real_tokens + generated_tokens) / wall_clock_s
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
                "num_decode_tokens": args.num_decode_tokens,
                "real_tokens": real_tokens,
                "total_tokens": total_tokens,
                "pad_tokens": pad_tokens,
                "generated_tokens": generated_tokens,
                "ttft_s": round(ttft_s, 6),
                "generation_time_s": round(generation_time_s, 6),
                "wall_clock_s": round(wall_clock_s, 6),
                "decode_tps": round(decode_tps, 2),
                "end_to_end_tps": round(end_to_end_tps, 2),
                "raw_tps": round(raw_tps, 2),
                "effective_tps": round(effective_tps, 2),
                "max_memory_gb": round(max_mem_gb, 3),
            }
            all_results.append(record)

            print_rank_0(
                f"  batch {batch_dict['batch_id']:02d} | "
                f"pad={batch_dict['actual_pad_ratio']:.3f} "
                f"cov={batch_dict['actual_cov']:.3f} | "
                f"prompt_real={real_tokens} prompt_total={total_tokens} "
                f"gen={generated_tokens} | "
                f"ttft={ttft_s:.3f}s gen={generation_time_s:.3f}s total={wall_clock_s:.3f}s | "
                f"decode_tps={decode_tps:.1f} e2e_tps={end_to_end_tps:.1f} | "
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
