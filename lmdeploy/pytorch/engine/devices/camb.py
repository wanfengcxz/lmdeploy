# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .base_device_utils import BaseDeviceUtils


class CAMBDeviceUtils(BaseDeviceUtils):

    device = 'camb'

    @classmethod
    def update_step_context(cls, step_context):
        """update step context."""
        kv_start_indices, attention_mask = [], []
        _, block_size, _, _ = step_context.kv_caches[0][0].shape
        for i in range(step_context.q_start_loc.size(0)):
            single_attention_mask = torch.logical_not(
                torch.tril(
                    torch.ones(step_context.q_seq_length[i],
                               step_context.block_offsets.shape[1] * block_size,
                               dtype=torch.bool).cuda(),
                    diagonal=step_context.kv_seq_length[i] -
                    step_context.q_seq_length[i],
                ))
            attention_mask.append(single_attention_mask)
            history_length = step_context.history_lengths[i]
            block_idx = history_length // block_size
            block_loc = step_context.block_offsets[i][block_idx]
            token_loc = history_length % block_size
            for _ in range(step_context.q_seq_length[i]):
                kv_start_indices.append([block_loc * block_size + token_loc])
                if _ == step_context.q_seq_length[i] - 1:
                    break
                token_loc = (token_loc + 1) % block_size
                block_idx = block_idx if token_loc else block_idx + 1
                block_loc = step_context.block_offsets[i][block_idx]
        kv_start_indices = torch.tensor(
            kv_start_indices, device=step_context.block_offsets.device)
        setattr(step_context, 'kv_start_indices', kv_start_indices)
        setattr(step_context, 'attention_mask', attention_mask)
        is_unpaged_prefill = (not step_context.is_decoding) and all(
            (step_context.q_seq_length == step_context.kv_seq_length).tolist())
        setattr(step_context, 'is_unpaged_prefill', is_unpaged_prefill)

        batch_size = step_context.q_start_loc.shape[0]
        cu_seq_lens = torch.zeros(batch_size+1, dtype=torch.int32, device=step_context.block_offsets.device)
        cu_seq_lens[:-1] = step_context.q_start_loc
        cu_seq_lens[-1] = step_context.q_seq_length.sum()
        setattr(step_context, 'cu_seq_lens', cu_seq_lens)

        return step_context

