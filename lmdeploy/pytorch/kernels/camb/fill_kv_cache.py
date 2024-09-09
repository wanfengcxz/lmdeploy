# Copyright (c) OpenMMLab. All rights reserved.
import infer_ext.ops as ext_ops
from torch import Tensor

def fill_kv_cache(
    key_states: Tensor,
    value_states: Tensor,
    key_caches: Tensor,
    value_caches: Tensor,
    q_start_loc: Tensor,
    q_seq_length: Tensor,
    kv_seq_length: Tensor,
    max_q_seq_length: int,
    block_offsets: Tensor,
    context: None,
):    
    """fill key/value state to cache for paged attention."""
   
    block_num, block_size, head_num, head_dim = key_caches.shape    # torch.Size([1644, 64, 8, 128])
    
    key_cache_reshaped = key_caches.view(key_caches.shape[0],key_caches.shape[2],key_caches.shape[1],key_caches.shape[3])
    value_cache_reshaped = value_caches.view(value_caches.shape[0],value_caches.shape[2],value_caches.shape[1],value_caches.shape[3])
    ext_ops.fill_kv_cache(key_states, value_states, key_cache_reshaped, value_cache_reshaped, context.kv_start_indices.view(context.kv_start_indices.shape[0]))


