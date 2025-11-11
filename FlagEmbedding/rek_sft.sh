export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0,1
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo


train_data="\
    /media/a822/82403B14403B0E83/Gwb/RAG/Train/train_data.jsonl "

# set large epochs and small batch size for testing
num_train_epochs=2
per_device_train_batch_size=1
gradient_accumulation_steps=4
train_group_size=8

# set num_gpus to 2 for testing
num_gpus=1

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

model_args="\
    --model_name_or_path /media/a822/82403B14403B0E83/Gwb/RAG/Rerank_model/bge-reranker-v2-m3 \
    --cache_dir $HF_HUB_CACHE \
"

data_args="\
    --train_data $train_data \
    --cache_path ~/.cache \
    --train_group_size $train_group_size \
    --query_max_len 256 \
    --passage_max_len 256 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation True \
"

training_args="\
    --output_dir /media/a822/82403B14403B0E83/Gwb/RAG/Rerank_model/bge-reranker-v2-m3_ft \
    --overwrite_output_dir \
    --learning_rate 2e-5 \
    --fp16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --weight_decay 0.01 \
    --deepspeed ./ds_stage0.json \
    --logging_steps 1 \
    --save_steps 1000 \
"

cmd="torchrun --nproc_per_node $num_gpus --master_port 12346 \
    -m FlagEmbedding.finetune.reranker.encoder_only.base \
    $model_args \
    $data_args \
    $training_args \
    > /media/a822/82403B14403B0E83/Gwb/RAG/logs/model_train.log 2>&1 &
"

echo $cmd
eval $cmd