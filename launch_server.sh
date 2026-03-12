#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Qwen3.5 MoE model path.
MODEL_PATH=/projects/models/qwen3_5_397B
LOG_FILE=server.log

# Basic runtime settings.
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=8
export DP_SIZE=4
export TP_SIZE=2
export ACT_TYPE=FP8
export LOAD_PYTHON_MODEL=1
export SEQ_SIZE_PER_BLOCK=2048
export MAX_SEQ_LEN=12800

# CUDA graph / low-latency settings from smoke_test.
export ENABLE_CUDA_GRAPH=0
# export DECODE_CAPTURE_CONFIG=1,2,3,4,5,6,7,8
export USE_DEEPEP_LOW_LATENCY=0
export WARM_UP=0
export NOT_USE_DEFAULT_STREAM=1
export ACCL_LOW_LATENCY_OPTIMIZE=1
export CONCURRENCY_LIMIT=8
export DEVICE_RESERVE_MEMORY_BYTES=-8192000000
export RESERVER_RUNTIME_MEM_MB=8192

# MoE / runtime compatibility flags from your reference command.
export MODEL_TYPE=qwen35_moe
export HACK_LAYER_NUM=4
# export QUANTIZATION=FP8_PER_CHANNEL_COMPRESSED
export USE_ALL_GATHER=1
export ROCM_DISABLE_CUSTOM_AG=True
export FT_DISABLE_CUSTOM_AR=True

export START_PORT=8088
export TOKENIZER_PATH=$MODEL_PATH
export CHECKPOINT_PATH=$MODEL_PATH
export FT_SERVER_TEST=1
export USE_ASM_PA=0
/opt/conda310/bin/python3.10 -m rtp_llm.start_server 2>&1 | tee $LOG_FILE