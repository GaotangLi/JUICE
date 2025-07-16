#!/usr/bin/env bash
set -x

# Define dataset and model lists properly
datasets="proverb_ending proverb_translation history_of_science_qa hate_speech_ending NQ_swap_dataset"
models="llamma2 llamma3 gemma phi2 stablelm olmo1"

# Loop over datasets and models
for dataset in $datasets; do 
    for model in $models; do 
        python ./script/MetaRun_Context.py \
            --model "$model" --dataset "$dataset" \
            --device_num "$Your_device_num"
    done 
done