#!/bin/sh

python videollama2/train.py \
    --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
    --data_path "/mnt/nfs/proj/hnl_downloaded_public_data/PFCTS/" \
    --text_path "/mnt/nfs/proj/hnl_downloaded_public_data/clip_description.csv" \
    --is_timeseries True \
    --output_dir "/home/barbosa/rihome/projects/TSLLaMA2/models" \
    --cache_dir "/home/barbosa/rihome/projects/TSLLaMA2/cache" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --fp16 True \
    --save_steps 1000 \
    --logging_steps 100
