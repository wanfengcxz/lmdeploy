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
    print(f"[rotary, q]: {query_states[0, 1, 0:100:10].cpu()}  {query_states.abs().mean()}")
    print(f"[rotary, k]: {key_states[0, 1, 0:100:10].cpu()}  {key_states.abs().mean()}")
    total_seqlen_q, head, dim = query_states.shape
    if not (hasattr(context, 'cos') or hasattr(context, 'sin')):
        if len(cos.shape) == 2 and len(sin.shape) == 2:
            cos_curr = cos[position_ids_1d]
            sin_curr = sin[position_ids_1d]
        else:
            raise RuntimeError("Cannot handle cos/sin shape dims!")

        if context:
            setattr(context, 'cos', cos_curr)
            setattr(context, 'sin', sin_curr)
    
    cached_cos = context.cos if context else cos
    cached_sin = context.sin if context else sin
    query_states, key_states = ext_ops.apply_rotary_pos_emb(query_states, key_states, cached_cos, cached_sin, position_ids_1d, cos, sin)
    if q_embed is None:
        q_embed = query_states
    else:
        q_embed.copy_(query_states)
    if k_embed is None:
        k_embed = key_states
    else:
        k_embed.copy_(key_states)
    print(f"[rotary, q_embed]: {q_embed[0, 1, 0:100:10].cpu()}  {q_embed.abs().mean()}")
    print(f"[rotary, k_embed]: {k_embed[0, 1, 0:100:10].cpu()}  {k_embed.abs().mean()}")
    return q_embed, k_embed
