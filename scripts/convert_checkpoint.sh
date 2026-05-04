singularity exec --nv --bind /home/jjimmy/SHELL.cmsc828:/workspace nemo_26.04.00.sif bash

salloc --nodes=1 --gres=gpu:a100:1 --time=01:00:00 --partition=gpu

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=qwen3-convert
#SBATCH --output=/scratch/zt1/project/cmsc828/user/jjimmy/project/logs/convert_%j.out
#SBATCH --error=/scratch/zt1/project/cmsc828/user/jjimmy/project/logs/convert_%j.err

PROJECT=/scratch/zt1/project/cmsc828/user/jjimmy/project
SIF=${PROJECT}/nemo_26.04.00.sif

mkdir -p ${PROJECT}/logs

singularity exec --nv \
    --bind ${PROJECT}:/workspace \
    ${SIF} \
    bash -c "
        cd /workspace/Megatron-LM/tools/checkpoint && \
        python convert.py \
            --model-type GPT \
            --loader llama_mistral \
            --saver mcore \
            --model-size qwen2.5 \
            --checkpoint-type hf \
            --target-tensor-parallel-size 4 \
            --load-dir /workspace/Qwen3-32B \
            --save-dir /workspace/qwen3-32b-mcore-tp4 \
            --tokenizer-model /workspace/Qwen3-32B \
            --bf16
    "
    
    
    
from megatron.bridge import AutoBridge

# 1) Create a bridge from a Hugging Face model (hub or local path)
bridge = AutoBridge.from_hf_pretrained("./Qwen3-32B", trust_remote_code=True)

# 2) Get a Megatron provider and configure parallelism before instantiation
provider = bridge.to_megatron_provider()
provider.tensor_model_parallel_size = 4
provider.pipeline_model_parallel_size = 1
provider.finalize()
# 3) Materialize Megatron Core model(s)
model = provider.provide_distributed_model(wrap_with_ddp=False)

# 4a) Export Megatron → Hugging Face (full HF folder with config/tokenizer/weights)
bridge.save_hf_pretrained(model, "./output")


singularity exec --nv \
      --bind /home/jjimmy/SHELL.cmsc828:/workspace \
      --writable-tmpfs \
      nemo_26.04.00.sif \
      bash


singularity exec --nv --writable --bind /home/jjimmy/SHELL.cmsc828:/workspace nemo_sandbox bash
