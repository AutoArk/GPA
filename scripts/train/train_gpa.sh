#!/bin/bash


MODEL="/path/to/gpa"
DATA="merged_shuffled_train.jsonl"
EVAL_DATA="merged_shuffled_train.jsonl"
DS_CONFIG_PATH="ds_config_zero2.json"
USE_LORA=False
Q_LORA=False
GLM_TOKENIZER_PATH="/path/to/gpa/glm-4-voice-tokenizer"
BICODEC_TOKENIZER_PATH="/path/to/gpa/BiCodec/"
OUTPUT_DIR="output_gpa"

deepspeed train_gpa.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --glm_tokenizer_path  $GLM_TOKENIZER_PATH \
    --bicodec_tokenizer_path $BICODEC_TOKENIZER_PATH \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 500 \
    --eval_strategy "steps" \
    --eval_steps 500 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0.005 \
    --adam_beta2 0.95 \
    --do_train \
    --warmup_ratio 0.005 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --model_max_length 1500 \
    --remove_unused_columns False \
    --use_lora ${USE_LORA} \
    --q_lora ${Q_LORA} \
    --gradient_checkpointing \
    --dataloader_pin_memory False \
    --deepspeed ${DS_CONFIG_PATH}