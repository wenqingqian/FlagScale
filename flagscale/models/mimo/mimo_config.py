# Copyright (c) 2025, BAAI. All rights reserved.

"""MIMO generic parallelism configuration."""

from dataclasses import dataclass


@dataclass
class ModuleParallelismConfig:
    """Parallelism configuration for a single MIMO module.

    All sizes must multiply to the world size for colocated deployment.
    """

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    data_parallel_size: int = 1
    context_parallel_size: int = 1


@dataclass
class MIMOParallelismConfig:
    """Colocated MIMO parallelism configuration for vision and language modules."""

    vision: ModuleParallelismConfig
    language: ModuleParallelismConfig

    def __post_init__(self):
        assert (
            self.vision.context_parallel_size == 1 and self.language.context_parallel_size == 1
        ), "Module context parallelism is restricted to 1 in colocated MIMO."


def compute_vit_batch_factor(
    vision_data_parallel_size: int,
    vision_micro_batch_size: int,
    language_data_parallel_size: int,
    language_micro_batch_size: int,
    num_microbatches: int,
) -> int:
    """Return the vit_batch_factor relating vision and language batching.

    ``vision_micro_batch_size`` (vbs) is the ViT-side micro batch size: the
    number of samples each vision DP entity processes in one forward, the
    vision analog of the LM ``micro_batch_size`` (mbs).  ``vit_batch_factor``
    (vbf) is the derived relation ``(vision_dp * vbs) / (language_dp * mbs)`` —
    how many LLM microbatches one ViT macro forward serves.  vbf == 1 selects
    the direct path (no scheduler).
    """
    vision_samples = vision_data_parallel_size * vision_micro_batch_size
    language_samples = language_data_parallel_size * language_micro_batch_size
    # Reject silent floor-division truncation (e.g. 16 // 12) that would
    # silently drop vision_micro_batch_size back to the direct path.
    assert vision_samples % language_samples == 0, (
        f"vision_dp * vision_micro_batch_size ({vision_samples}) must be a multiple of "
        f"language_dp * micro_batch_size ({language_samples}). "
        f"Adjust vision_micro_batch_size or micro_batch_size."
    )
    vit_batch_factor = vision_samples // language_samples
    assert num_microbatches % vit_batch_factor == 0, (
        f"num_microbatches ({num_microbatches}) must be a multiple of "
        f"vit_batch_factor ({vit_batch_factor}) for MIMO correctness. "
        f"Adjust global_batch_size, micro_batch_size, or vision_micro_batch_size."
    )
    return vit_batch_factor
