#!/bin/bash

model="internlm2"

if [ ${model} == "llama2" ];then
    ## llama2 7B
    tp=1
    max_batch_size=8
    cache_max_entry_count=0.4
    model_path="/data2/share_data/llama_model_data/llama-2-7b-chat-hf"
    result_file="llama2_pt_7b_thr.csv"
elif [ ${model} == "internlm2" ];then
    ## internlm2 7B
    tp=1
    max_batch_size=8
    cache_max_entry_count=0.4
    model_path="/root/.cache/modelscope/hub/Shanghai_AI_Laboratory/internlm2-chat-7b"
    result_file="internlm2_pt_7b_thr.csv"
elif [ ${model} == "mixtral" ];then
    ## Mixtral-8x7B
    tp=2
    max_batch_size=8
    cache_max_entry_count=0.4
    model_path="/data2/share_data/mixtral_model_data/Mixtral-8x7B-Instruct-v0.1"
    result_file="Mixtral-8x7B_pt_7b_thr.csv"
fi

python3 profile_generation.py ${model_path} --backend pytorch --tp ${tp} --concurrency ${max_batch_size} --cache-max-entry-count ${cache_max_entry_count} --prompt-tokens 256 --completion-tokens 128 --test-round 3 --warmup-round 1 --csv ${result_file}
