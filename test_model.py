import torch
import lmdeploy
from lmdeploy.vl import load_image
from lmdeploy import PytorchEngineConfig

import dlinfer
from modelscope import HubApi

PIC1 = '/data/wangqing/work/1023/lmdeploy/tiger.jpeg'
PIC2 = '/data/wangqing/work/1023/lmdeploy/xiaoxin.jpg'

def chat(pipe):
    sess = pipe.chat("Hi, pls intro yourself")
    print("------first session: ", sess)
    sess = pipe.chat('What are you good at?', session=sess)
    print("------second session: ", sess)

def question(pipe):
    #question = ["Hi, pls intro yourself"]
    question = ["Hi, pls intro yourself", "Hi, pls intro yourself in detail"]
    response = pipe(question, do_preprocess=False, top_k=1)
    print(response)

def question_img(pipe):
    image = load_image(PIC1)
    response = pipe(("describe this image", image))
    print(response)


def test_model(model_name, tp_list, eager_mode_list, is_vlm):
    for tp in tp_list:
        for eager_mode in eager_mode_list:

            backend = PytorchEngineConfig(tp=tp,block_size=16, cache_max_entry_count=0.4, \
                device_type="camb", download_dir="/data/share/llm_model", eager_mode=eager_mode)
            pipe = lmdeploy.pipeline(model_name, backend_config = backend)

            if is_vlm:    
                question_img(pipe)
            else:
                question(pipe)

            print(f"{model_name} tp={tp} eager_mode={eager_mode} test success!")

if __name__ == "__main__":
    
    # torch.set_printoptions(precision=10)
    
    api=HubApi()
    api.login('a1cfd264-4de3-435b-9d68-085bc4762c4a')

    #model_name = "Shanghai_AI_Laboratory/internlm2_5-7b"
    #model_name = "Shanghai_AI_Laboratory/internlm2-chat-7b"
    #model_name = "OpenGVLab/InternVL2-2B"
    #model_name = "AI-ModelScope/InternVL-Chat-V1-5"
    #model_name = "LLM-Research/Meta-Llama-3-8B"
    #model_name = "AI-ModelScope/Mixtral-8x7B-v0.1"
    #model_name = "Qwen/Qwen2-7B"
    #model_name = "Qwen/Qwen2-57B-A14B"
    #model_name = "ZhipuAI/cogvlm2-llama3-chinese-chat-19B"

    #model_name = "shakechen/Llama-2-7b-hf"
    
    tp_list = [1, 2]
    #eager_mode_list = [False]
    eager_mode_list = [True]
    #test_model("Shanghai_AI_Laboratory/internlm2_5-7b", tp_list, eager_mode_list, is_vlm=False)
    #test_model("Shanghai_AI_Laboratory/internlm2-chat-7b", tp_list, eager_mode_list, is_vlm=False)
    test_model("OpenGVLab/InternVL2-2B", tp_list, eager_mode_list, is_vlm=True)


