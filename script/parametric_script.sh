#!/usr/bin/env bash
set -x

# Define dataset and model lists properly
datasets="company_founder book_author official_language world_capital company_headquarter athlete_sport"
models="llamma2 llamma3 gemma phi2 stablelm olmo1"

# Loop over datasets and models
for dataset in $datasets; do 
    for model in $models; do 
        python ./script/MetaRun_Parametric.py \
            --model "$model" --dataset "$dataset" \
            --device_num "$Your_device_num"
    done 
done