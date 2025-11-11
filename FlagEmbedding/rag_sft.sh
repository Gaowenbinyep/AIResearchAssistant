#!/bin/bash

# 选择物理 GPU
export CUDA_VISIBLE_DEVICES=0
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
# 启动分布式训练
torchrun --nproc_per_node 1 --master_port 12346 \
	-m FlagEmbedding.finetune.embedder.encoder_only.m3 \
	--model_name_or_path ../embedding_model/bge-m3 \
    --cache_dir ../embedding_model/cache \
    --train_data ../Train/train_data.jsonl \
    --cache_path ../Train \
    --train_group_size 8 \
    --query_max_len 64 \
    --passage_max_len 2600 \
    --pad_to_multiple_of 8 \
    --same_dataset_within_batch True \
    --small_threshold 0 \
    --drop_threshold 0 \
    --output_dir ../embedding_model/bge-m3-ft \
    --overwrite_output_dir \
    --learning_rate 5e-6 \
    --fp16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ./ds_stage0.json \
    --logging_steps 1 \
    --save_steps 500 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type m3_kd_loss \
    --unified_finetuning True \
    --use_self_distill True \
    --fix_encoder False \
    --self_distill_start_step 0 \
    > /media/a822/82403B14403B0E83/Gwb/RAG/logs/model_train.log 2>&1 &