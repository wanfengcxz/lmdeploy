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
    total_seqlen_q, head, dim = query_states.shape
    if not (hasattr(context, 'cos') or hasattr(context, 'sin')):
        if len(cos.shape) == 2 and len(sin.shape) == 2:
            cos = cos[position_ids_1d]
            sin = sin[position_ids_1d]
        else:
            raise RuntimeError("Cannot handle cos/sin shape dims!")

        if context:
            setattr(context, 'cos', cos)
            setattr(context, 'sin', sin)
    
    cached_cos = context.cos if context else cos
    cached_sin = context.sin if context else sin
    ext_ops.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids_1d, cached_cos, cached_sin)
    if q_embed is None:
        q_embed = query_states
    else:
        q_embed.copy_(query_states)
    if k_embed is None:
        k_embed = key_states
    else:
        k_embed.copy_(key_states)

    return q_embed, k_embed
