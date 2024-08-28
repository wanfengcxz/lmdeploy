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
    pass
