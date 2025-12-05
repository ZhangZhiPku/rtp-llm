bazelisk build //rtp_llm:rtp_llm --config=cuda12_6

export PYTHONPATH=~/work/RTP-LLM/bazel-out/k8-opt/bin/:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="2,3"
export TP_SIZE=2
export DP_SIZE=1
export WORLD_SIZE=2
export EP_SIZE=2
export START_PORT=26000
export MODEL_TYPE=qwen_3_moe # 根据需要设置
export USE_RPC_MODEL=1
export MAX_SEQ_LEN=5120
export ENABLE_FMHA=on
export WARM_UP=0
export CHECKPOINT_PATH=~/hf/Qwen3-30B-A3B/
export TOKENIZER_PATH=~/hf/Qwen3-30B-A3B/
export LOG_LEVEL=INFO # 这里我喜欢打开TRACE，你随意
export NCCL_DISABLE_ABORT=1
export FT_DISABLE_CUSTOM_AR=1
export CUDA_LAUNCH_BLOCKING=0
export NOT_USE_DEFAULT_STREAM=True
export DEVICE_RESERVE_MEMORY_BYTES=-16048000000
#export ACT_TYPE=BF16
export RESERVER_RUNTIME_MEM_MB=8048
export ENABLE_COMM_OVERLAP=0

/opt/conda310/bin/python3 -m rtp_llm.start_server