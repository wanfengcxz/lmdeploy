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
    # key_states torch.Size([144, 32, 128])
    # key_caches torch.Size([1198, 64, 32, 128])
    # q_start_loc torch.Size([1])
    # q_seq_length torch.Size([1])
    # kv_seq_length torch.Size([1])
    # max_q_seq_length 144
    # block_offsets torch.Size([1, 3])
    # context.kv_start_indices torch.Size([144, 1])
    """fill key/value state to cache for paged attention."""
    #print("k:",key_states.shape)
    #print("kc:",key_caches.shape)
    #if key_caches.shape[1] != key_states.shape[1]:
    #    key_caches = key_caches.reshape(key_caches.shape[0],key_caches.shape[2],key_caches.shape[1],key_caches.shape[3])
    #    value_caches = value_caches.reshape(value_caches.shape[0],value_caches.shape[2],value_caches.shape[1],value_caches.shape[3])
    #print("kc after:",key_caches.shape)
    
    block_num, block_size, head_num, head_dim = key_caches.shape    # torch.Size([1644, 64, 8, 128])
    print(f"[fill_kv_cache, k_cache shape]: {key_caches.shape}")
    print(f"[fill_kv_cache, k_index]: {context.kv_start_indices.flatten()}")  # torch.Size([9, 1])    torch.Size([1, 1])
    print(f"[fill_kv_cache, k]: {key_states[0, 0, 0:100:10].cpu()}  {key_states.abs().mean().cpu()}")

    # key_cache_reshaped = key_caches.permute(0, 2, 1, 3).contiguous()
    # value_cache_reshaped = value_caches.permute(0, 2, 1, 3).contiguous()
    key_cache_reshaped = key_caches.view(key_caches.shape[0],key_caches.shape[2],key_caches.shape[1],key_caches.shape[3])
    value_cache_reshaped = value_caches.view(value_caches.shape[0],value_caches.shape[2],value_caches.shape[1],value_caches.shape[3])
    ext_ops.fill_kv_cache(key_states, value_states, key_cache_reshaped, value_cache_reshaped, context.kv_start_indices.view(context.kv_start_indices.shape[0]))
   #  bt_ops.reshape_paged_cache(key, value, key_cache_reshaped, value_cache_reshaped, kv_indices)
    # key_caches[...] = key_cache_reshaped.permute(0, 2, 1, 3).contiguous()
    # value_caches[...] = value_cache_reshaped.permute(0, 2, 1, 3).contiguous()
    
    if context.kv_start_indices.shape[0] != 1:
        # prefill
        print(f"[after fill_kv_cache, k_cache]: {key_caches[0, 0, 0, 0:100:10].cpu()}  {key_caches.abs().mean().cpu()}")
    else:
        # decoder
        k_idx = context.kv_start_indices[0][0]
        block_id = k_idx // block_size
        block_offset = k_idx % block_size
        print(f"[after fill_kv_cache, k_cache]: {key_caches[block_id, block_offset, 0, 0:100:10].cpu()}  {key_caches.abs().mean().cpu()}")


    # ext_ops.fill_kv_cache(key_states, value_states, key_caches, value_caches, context.kv_start_indices.view(context.kv_start_indices.shape[0]))
    # ext_ops.fill_kv_cache(key_states, value_states, key_caches.view(value_caches.shape[0],value_caches.shape[2],value_caches.shape[1],value_caches.shape[3]), value_caches.view(value_caches.shape[0],value_caches.shape[2],value_caches.shape[1],value_caches.shape[3]), context.kv_start_indices.view(context.kv_start_indices.shape[0]))
