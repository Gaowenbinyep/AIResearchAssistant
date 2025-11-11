# 后台下载并将进度输出到日志文件
nohup modelscope download \
  --model BAAI/bge-reranker-v2-m3 \
  --local_dir ./bge-reranker-v2-m3 \
  > ../logs/download_progress.log 2>&1 &
