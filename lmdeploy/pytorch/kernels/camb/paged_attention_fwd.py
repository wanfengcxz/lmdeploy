# Copyright (c) OpenMMLab. All rights reserved.
import infer_ext.ops as ext_ops
import torch
from torch import Tensor


def flash_context_attention(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    attn_output: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_offsets: Tensor,
    q_start_loc: Tensor,
    q_seq_len: Tensor,
    kv_seq_len: Tensor,
    block_size: int,
    kv_cache_len: int,
    context=None,
):
    num_q_heads = query_states.shape[1]
    num_kv_heads = key_states.shape[1]
    output = torch.empty_like(query_states)
    ext_ops.context_attention(
            query_states,
            key_states,
            value_states,
            q_start_loc,
            q_seq_len,
            context.max_q_seq_length,
            num_q_heads,
            num_kv_heads,
            #attn_mask=context.attention_mask,
            attn_output = output,
            cu_seq_lens = context.cu_seq_lens,
            max_seq_len = context.max_kv_seq_length,
            #attn_output=attn_output.view(query_states.shape),
        )
    attn_output.copy_(output)

def paged_token_attention(q, k_cache, v_cache, attn_output, kv_seq_len,
                          max_kv_seq_len, block_offsets, block_size):

    num_kv_heads, num_q_heads = k_cache.shape[1], q.shape[1]
    if q.ndim == 3:
        q = q.unsqueeze(1)

    ext_ops.paged_decode_attention(
        q,
        k_cache,
        v_cache,
        block_offsets,
        block_size,
        kv_seq_len,
        num_q_heads,
        num_kv_heads,
        attn_output = attn_output.view(q.shape),
        max_context_lens = max_kv_seq_len,
    )


def paged_attention_fwd(
    query_states: Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    attn_output: Tensor,
    block_offsets: Tensor,
    q_start_loc: Tensor,
    q_seqlens: Tensor,
    kv_seqlens: Tensor,
    max_seqlen: int,
    window_size: int = 1,
    context=None,
):
    is_decoding = query_states.shape[-3] == q_seqlens.size(0) 
    block_num, block_size, head_num, head_size = key_cache.size()
    #k = key_cache.view(block_num, head_num, block_size, head_size)
    #v = value_cache.view(block_num, head_num, block_size, head_size)
    k = key_cache
    v = value_cache
    kv_cache_len = block_num * block_size
    if not is_decoding:
        flash_context_attention(
            query_states,
            key_states,
            value_states,
            attn_output,
            k,
            v,
            block_offsets,
            q_start_loc,
            q_seqlens,
            kv_seqlens,
            block_size,
            kv_cache_len,
            context=context,
        )
       
    else:
        paged_token_attention(
            query_states,
            k,
            v,
            attn_output,
            kv_seqlens,
            context.max_kv_seq_length,
            block_offsets,
            block_size,
        )
      
