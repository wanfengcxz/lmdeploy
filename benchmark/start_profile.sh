#!/bin/bash

model=internlm2

if [ ${model} == "llama2" ];then
    ## llama2 7B
    tp=1
    max_batch_size=128
    cache_max_entry_count=0.4
    model_path="/data2/share_data/llama_model_data/llama-2-7b-chat-hf"
    result_file="llama2_pt_7b_thr.csv"
elif [ ${model} == "internlm2" ];then
    ## internlm2 7B
    tp=1
    max_batch_size=128
    cache_max_entry_count=0.4
    model_path="/data/share/llm_model/internlm2-chat-7b"
    result_file="internlm2_pt_7b_thr.csv"
elif [ ${model} == "mixtral" ];then
    ## Mixtral-8x7B
    tp=2
    max_batch_size=128
    cache_max_entry_count=0.4
    model_path="/data2/share_data/mixtral_model_data/Mixtral-8x7B-Instruct-v0.1"
    result_file="Mixtral-8x7B_pt_7b_thr.csv"
elif [ ${model} == "internvl" ];then
    ## Mixtral-8x7B
    tp=4
    max_batch_size=128
    cache_max_entry_count=0.3
    model_path="/data2/share_data/internvl_model_data/InternVL-Chat-V1-5"
    result_file="InternVL_pt_thr.csv"
fi

python3 profile_throughput.py ShareGPT_V3_unfiltered_cleaned_split.json ${model_path} --backend pytorch --tp ${tp} --concurrency ${max_batch_size} --cache-max-entry-count ${cache_max_entry_count} --csv ${result_file}
