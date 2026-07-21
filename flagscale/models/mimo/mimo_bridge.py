# Copyright (c) 2025, BAAI. All rights reserved.

"""Colocated cross-rank utilities for MIMO module communication.

In colocated mode rank sets overlap, so inter-module communication can use
intra-node P2P/broadcast instead of the full ``BridgeCommunicator``.  The
helpers here assume the deterministic rank ordering produced by
``flagscale.models.mimo.hetero_pg_utils``.
"""

import torch.distributed as dist


def _get_language_tp_first_global_rank(language_pg):
    """Return the global rank of the first member of the language TP group."""
    if language_pg is None or language_pg.tp is None:
        return dist.get_rank()
    # Query group membership instead of assuming contiguous TP ranks.
    return dist.get_process_group_ranks(language_pg.tp)[0]


def get_source_vision_rank(language_pg, forward_idx_in_round, vit_batch_factor):
    """Return the owner ViT rank of the given forward's microbatch slice.

    Within a language TP group, every member is also a ViT rank because vision
    TP size is 1.  The group's macro batch is split into whole microbatches
    (see ``Qwen35MIMOModel._my_microbatch_range``): TP member j computes and
    supplies the microbatches in its slice, so this mapping is the actual data
    ownership, not a load-balancing choice.

    Example (world size 8, language TP=2, vit_batch_factor=2):
      - language TP group {2, 3} has ViT ranks 2 and 3.
      - forward 0 in the round belongs to ViT rank 2; forward 1 to ViT rank 3.

    Example (language TP=2, vit_batch_factor=4):
      - Each ViT rank owns enough samples for two LLM forwards.
      - forwards 0,1 belong to rank 2; forwards 2,3 to rank 3.
    """
    assert forward_idx_in_round >= 0, (
        f"forward_idx_in_round must be non-negative, got {forward_idx_in_round}"
    )
    assert vit_batch_factor >= 1, f"vit_batch_factor must be >= 1, got {vit_batch_factor}"
    tp_first = _get_language_tp_first_global_rank(language_pg)
    language_tp_size = max(1, dist.get_world_size(language_pg.tp))
    if vit_batch_factor == 1:
        # Direct path (no scheduler): every LLM forward reuses the TP-first
        # rank's ViT output; offset is always 0, no divisibility requirement.
        return tp_first
    # Guard against source-rank cycling that would leave the current TP group.
    assert vit_batch_factor % language_tp_size == 0, (
        f"vit_batch_factor ({vit_batch_factor}) must be divisible by language_tp_size "
        f"({language_tp_size}) for colocated MIMO source-rank cycling"
    )
    offset = (forward_idx_in_round * language_tp_size // vit_batch_factor) % language_tp_size
    src_rank = tp_first + offset
    assert src_rank >= 0, f"computed source vision rank {src_rank} is negative"
    return src_rank


def broadcast_to_language_tp(tensor, language_pg, src_rank):
    """Broadcast ``tensor`` from ``src_rank`` to all ranks in the language TP group.

    Args:
        tensor: Tensor on the current rank.  Only meaningful on ``src_rank``
            before the call; after the call every rank in the TP group holds
            the same data.
        language_pg: Language module process group collection.
        src_rank: Global rank of the source within the language TP group.

    Returns:
        Tensor after broadcast.
    """
    if language_pg is None or language_pg.tp is None:
        return tensor
    if dist.get_world_size(language_pg.tp) <= 1:
        return tensor
    tensor = tensor.contiguous()
    dist.broadcast(tensor, src=src_rank, group=language_pg.tp)
    return tensor


def reduce_visual_grad_from_language_tp(grad, language_pg, dst_rank):
    """Reduce gradients from all language TP ranks back to ``dst_rank``.

    When a visual embed has been broadcast to the language TP group, each TP
    rank accumulates its own gradient during LLM backward.  This helper sums
    those gradients and places the result on ``dst_rank``.
    """
    if language_pg is None or language_pg.tp is None:
        return grad
    if dist.get_world_size(language_pg.tp) <= 1:
        return grad
    grad = grad.contiguous()
    dist.reduce(grad, dst=dst_rank, group=language_pg.tp, op=dist.ReduceOp.SUM)
    return grad
