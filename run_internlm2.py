import infer_ext
import lmdeploy
import torch
from lmdeploy import PytorchEngineConfig
if __name__ == "__main__":
    torch.set_printoptions(precision=10)
    pipe = lmdeploy.pipeline("/root/.cache/modelscope/hub/Shanghai_AI_Laboratory/internlm2-chat-7b",
                             backend_config = PytorchEngineConfig(tp=1,block_size=16, cache_max_entry_count=0.3, device_type="camb"))
    question = ["Hi, pls intro yourself in detail"]
    print(question)
    response = pipe(question, do_preprocess=False, top_k=1)
    print(response)
    # for idx, r in enumerate(response):
    #     print(f"Q: {question[idx]}")
    #     print(f"AAAAAA: {r.text}")
    #     print()
    # print("end")
