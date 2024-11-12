#!/bin/bash
model_path=/data/share/llm_model/Shanghai_AI_Laboratory/internlm2_5-7b
lmdeploy serve api_server ${model_path} --max-batch-size 128 --backend pytorch --device camb --model-name target --cache-block-seq-len 16
