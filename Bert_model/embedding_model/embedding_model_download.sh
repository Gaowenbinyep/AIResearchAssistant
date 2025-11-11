# 后台下载并将进度输出到日志文件
nohup modelscope download \
  --model BAAI/bge-m3 \
  --local_dir ./bge-m3 \
  > ../logs/download_progress.log 2>&1 &
