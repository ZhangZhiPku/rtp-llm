export HIP_VISIBLE_DEVICES=2,3
export SEQ_SIZE_PER_BLOCK=1024 
export KERNEL_SEQ_SIZE_PER_BLOCK=16 
export WARM_UP=0 
export ENABLE_CUDA_GRAPH=0
export DECODE_CAPTURE_CONFIG=1,2

export CONCURRENCY_LIMIT=4096
export LOAD_PYTHON_MODEL=1
export USE_ASM_PA=0
export WORLD_SIZE=8
export DP_SIZE=1
export TP_SIZE=8
export EP_SIZE=1
export DEVICE_RESERVE_MEMORY_BYTES=-8048000000     
export RESERVER_RUNTIME_MEM_MB=10240
export MAX_SEQ_LEN=262144
export START_PORT=8000
export ACT_TYPE=bf16
export TOKENIZER_PATH=/zhangzhi/models/qwen3_5_397B
export CHECKPOINT_PATH=/zhangzhi/models/qwen3_5_397B
export MODEL_TYPE=qwen35_moe
export FT_SERVER_TEST=1
export ROCM_DISABLE_CUSTOM_AG=True
export FT_DISABLE_CUSTOM_AR=True
export AITER_ASM_DIR=$(bazelisk info output_base)/external/aiter/aiter_meta/hsa/gfx950/
/opt/conda310/bin/python3.10 -m rtp_llm.start_server 2>&1 | tee server.log