# Runtime env shared by hangang SCOUT jobs (sourced after the scout_env.sqfs mount at /tmp/scout_env).
U=/scratch/t512a04/users/doyoon
# driver 570 = CUDA 12.8; torch cu129 / vLLM kernels need the 12.9 forward-compat libcuda (575)
export LD_LIBRARY_PATH=/apps/NVIDIA/hpc_sdk/Linux_aarch64/26.3/cuda/12.9/compat:${LD_LIBRARY_PATH:-}
# NCCL over Slingshot (CXI): aws-ofi-nccl plugin + libfabric settings from the aws-ofi-nccl/1.20.0 module
export LD_LIBRARY_PATH=/apps/library/aws-ofi-nccl/1.20.0/aarch64/lib:/opt/cray/libfabric/2.3.1/lib64:$LD_LIBRARY_PATH
export NCCL_NET="AWS Libfabric" NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 FI_PROVIDER=cxi
export FI_MR_CACHE_MONITOR=userfaultfd FI_CXI_DISABLE_HOST_REGISTER=1 FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_RDZV_PROTO=alt_read FI_CXI_RDZV_EAGER_SIZE=0 FI_CXI_RDZV_THRESHOLD=0 FI_CXI_RDZV_GET_MIN=0
export FI_CXI_DEFAULT_TX_SIZE=2048 FI_CXI_RX_MATCH_MODE=hybrid
export PATH=/tmp/scout_env/bin:$PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/tmp/hf_home TRITON_CACHE_DIR=/tmp/triton TORCHINDUCTOR_CACHE_DIR=/tmp/inductor VLLM_CACHE_ROOT=/tmp/vllm_cache TMPDIR=/tmp
export ELSA_ATTN_IMPL=sdpa VLLM_USE_V1=0 VLLM_HOST_IP=127.0.0.1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN WANDB_MODE=disabled
export VLLM_SERVER_LOG_DIR=$U/logs TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
