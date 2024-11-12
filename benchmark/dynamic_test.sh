#!/bin/bash

model_path=/data/share/llm_model/Shanghai_AI_Laboratory/internlm2_5-7b
#backend has to be one of [lmdeploy, vlim]
backend=vllm

#run the test script
for ((rate=6;rate<=6;rate+=4));do
  python /workspace/vllm-v0.4.2/benchmarks/benchmark_serving.py \
    --backend ${backend} \
    --host 0.0.0.0 --port 23333 \
    --dataset-path /data/share/llm_model/ShareGPT_V3_unfiltered_cleaned_split.json \
    --model target \
    --tokenizer ${model_path} \
    --num-prompts 1000 \
    --disable-tqdm \
    --request-rate ${rate}
done  
echo "+++++++++++++++++++++++++"

# for ((rate=6;rate<=6;rate+=4));do
#   python /workspace/vllm-v0.4.2/benchmarks/benchmark_serving.py \
#     --backend ${backend} \
#     --host 0.0.0.0 --port 23333 \
#     --dataset-path /workspace/volume/shangda/share/ShareGPT_V3_unfiltered_cleaned_split.json \
#     --model target \
#     --tokenizer ${model_path} \
#     --num-prompts 1000 \
#     --disable-tqdm \
#     --request-rate ${rate}
#   echo "!!!!!!!!!!!!!!!!!!!!!!"
# done
# echo "---------------------"

