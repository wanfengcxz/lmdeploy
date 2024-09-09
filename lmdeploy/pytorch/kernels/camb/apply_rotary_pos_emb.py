# Copyright (c) OpenMMLab. All rights reserved.
import infer_ext.ops as ext_ops
from torch import Tensor


def apply_rotary_pos_emb(
    query_states: Tensor,
    key_states: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Tensor,
    position_ids_1d: Tensor,
    q_embed=None,
    k_embed=None,
    context=None,
):
    cu_seq_lens = context.cu_seq_lens
    total_seqlen_q, head, dim = query_states.shape
    if not (hasattr(context, 'cos') or hasattr(context, 'sin')):
        assert cos.ndim == 2 and sin.ndim == 2, "camb only support 2-d cos and sin"
        if context:
            setattr(context, 'cos', cos)
            setattr(context, 'sin', sin)
    
    query_states, key_states = ext_ops.apply_rotary_pos_emb(query_states, key_states, None, None, position_ids_1d, cos, sin, cu_seq_lens)
    if q_embed is None:
        q_embed = query_states
    else:
        q_embed.copy_(query_states)
    if k_embed is None:
        k_embed = key_states
    else:
        k_embed.copy_(key_states)
    return q_embed, k_embed
