# Copyright (c) OpenMMLab. All rights reserved.
import infer_ext.ops as ext_ops
from torch import Tensor

def rms_norm(hidden_states: Tensor, weight: Tensor, epsilon: float = 1e-6):
    # hidden_states [total_seq_len, hidden_size]
    # weight [hidden_size]
    # print("rms")
    # print(f"hidden_states shape: {hidden_states.shape}")
    # print(f"weight shape: {weight.shape}")
    # print()
    return hidden_states
    # return ext_ops.rms_norm(hidden_states, weight, epsilon)
