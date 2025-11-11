#!/bin/bash
# 下载 Hugging Face 模型到本地，并将进度输出到日志文件
# 使用方式: ./download_model.sh

# 模型名称
MODEL_NAME="RJuro/SciNERTopic"
# 本地保存路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOCAL_DIR="${SCRIPT_DIR}/SciNERTopic"
# 日志文件
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/download_progress.log"

# 创建日志目录（如果不存在）
mkdir -p "${LOG_DIR}" "${LOCAL_DIR}"

# 如果网络不通，可以启用国内镜像 (清华 TUNA)
export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}

# 后台下载
nohup hf download "$MODEL_NAME" --local-dir "$LOCAL_DIR" \
  > "$LOG_FILE" 2>&1 &

echo "✅ 模型下载已开始：$MODEL_NAME"
echo "📂 保存路径: $LOCAL_DIR"
echo "📝 日志文件: $LOG_FILE"
echo "🔍 使用 'tail -f $LOG_FILE' 查看实时进度"
