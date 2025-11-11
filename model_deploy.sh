#!/usr/bin/env bash
set -Eeuo pipefail

# 仅在需要时导出，避免污染父环境
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
MODEL_PATH="${PROJECT_ROOT}/LLM/Qwen3-8B"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

# 如果需要外网访问，把 --host 设为 0.0.0.0
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve "${MODEL_PATH}" \
  --tensor-parallel-size 1 \
  --port 8888 \
  --disable-log-requests \
  --max-model-len 20000 \
  > "${LOG_DIR}/model_output.log" 2>&1 &
echo "vLLM 已后台启动，日志：${LOG_DIR}/model_output.log"
