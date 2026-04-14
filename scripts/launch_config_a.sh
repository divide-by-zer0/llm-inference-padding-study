#!/usr/bin/env bash
# Config A: TP=4, PP=1
# 4 GPUs, single pipeline stage, maximal tensor split.
# Run from the study root (llm-inference-padding-study/):
#   cd /path/to/llm-inference-padding-study
#   bash scripts/launch_config_a.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — edit these before running
# ---------------------------------------------------------------------------
CHECKPOINT="${CHECKPOINT:-/path/to/qwen3-32b-mcore-tp4}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/path/to/Qwen3-32B}"
BENCHMARK_DATASET_DIR="${BENCHMARK_DATASET_DIR:-../benchmark_dataset}"
RESULTS_DIR="${RESULTS_DIR:-../results}"

# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------
TP=4
PP=1
N_GPUS=4   # TP × PP

# ---------------------------------------------------------------------------
# Qwen3-32B architecture
# ---------------------------------------------------------------------------
NUM_LAYERS=64
HIDDEN_SIZE=5120
NUM_ATTN_HEADS=64
NUM_QUERY_GROUPS=8      # GQA: 8 KV heads
FFN_HIDDEN_SIZE=25600
MAX_SEQ_LEN=4096
VOCAB_SIZE=151936
ROTARY_BASE=1000000     # Qwen3 uses 1 M RoPE base

# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1

# Make gpt_builders / model_provider importable (they live in Megatron-LM root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_ROOT="${SCRIPT_DIR}/../Megatron-LM"
export PYTHONPATH="${MEGATRON_ROOT}:${PYTHONPATH:-}"

DISTRIBUTED_ARGS=(
    --nproc_per_node ${N_GPUS}
    --nnodes 1
    --node_rank 0
    --master_addr localhost
    --master_port 6001
)

torchrun "${DISTRIBUTED_ARGS[@]}" \
    scripts/run_padding_benchmark.py \
    \
    --tensor-model-parallel-size  ${TP} \
    --pipeline-model-parallel-size ${PP} \
    \
    --num-layers          ${NUM_LAYERS} \
    --hidden-size         ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTN_HEADS} \
    --group-query-attention \
    --num-query-groups    ${NUM_QUERY_GROUPS} \
    --ffn-hidden-size     ${FFN_HIDDEN_SIZE} \
    --seq-length          ${MAX_SEQ_LEN} \
    --max-position-embeddings ${MAX_SEQ_LEN} \
    \
    --swiglu \
    --normalization RMSNorm \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --rotary-base ${ROTARY_BASE} \
    --use-rotary-position-embeddings \
    \
    --transformer-impl transformer_engine \
    --attention-softmax-in-fp32 \
    --no-masked-softmax-fusion \
    \
    --bf16 \
    --load "${CHECKPOINT}" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "${TOKENIZER_MODEL}" \
    \
    --inference-max-requests 64 \
    --inference-max-seq-length ${MAX_SEQ_LEN} \
    \
    --benchmark-dataset-dir "${BENCHMARK_DATASET_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --config-name config_a \
    --n-warmup 2 \
    \
    --model-provider gpt \
    --micro-batch-size 1
