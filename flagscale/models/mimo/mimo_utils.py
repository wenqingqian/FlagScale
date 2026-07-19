# Copyright (c) 2025, BAAI. All rights reserved.

"""Generic tensor utilities for MIMO module communication."""

import torch
import torch.distributed as dist


def reshard_between_tp(
    tensor: torch.Tensor,
    src_pg,
    dst_pg,
    dim: int = 0,
) -> torch.Tensor:
    """Reshard a tensor from one TP group to another along ``dim``.

    Args:
        tensor: Local tensor shard on the source TP group.
        src_pg: Source tensor-parallel process group (or None/singleton).
        dst_pg: Destination tensor-parallel process group (or None/singleton).
        dim: Dimension along which the tensor is sharded. Default is 0, which
            matches both the ViT output layout ``[T, H]`` and the language
            embedding SP sharding.

    Returns:
        Local tensor shard on the destination TP group.
    """
    src_size = dist.get_world_size(src_pg) if src_pg is not None else 1
    dst_size = dist.get_world_size(dst_pg) if dst_pg is not None else 1

    assert src_size >= 1 and dst_size >= 1, (
        f"src_size={src_size} and dst_size={dst_size} must both be >= 1"
    )
    assert -tensor.ndim <= dim < tensor.ndim, (
        f"dim={dim} out of range for tensor with {tensor.ndim} dimensions"
    )

    if src_size == dst_size:
        return tensor

    # Gather the full tensor on the source TP group.
    if src_size > 1:
        gathered = [torch.empty_like(tensor) for _ in range(src_size)]
        dist.all_gather(gathered, tensor.contiguous(), group=src_pg)
        full_tensor = torch.cat(gathered, dim=dim)
    else:
        full_tensor = tensor

    # Slice for the destination TP group.
    if dst_size > 1:
        dst_rank = dist.get_rank(dst_pg) if dst_pg is not None else 0
        total = full_tensor.shape[dim]
        assert dst_size <= total, (
            f"destination TP size {dst_size} cannot exceed tensor size {total} along dim {dim}"
        )
        chunk_size = total // dst_size
        remainder = total % dst_size
        # Distribute remainder across first ``remainder`` ranks.
        if dst_rank < remainder:
            start = dst_rank * (chunk_size + 1)
            end = start + chunk_size + 1
        else:
            start = remainder * (chunk_size + 1) + (dst_rank - remainder) * chunk_size
            end = start + chunk_size
        slices = [slice(None)] * full_tensor.ndim
        slices[dim] = slice(start, end)
        return full_tensor[tuple(slices)].contiguous()

    return full_tensor.contiguous()
