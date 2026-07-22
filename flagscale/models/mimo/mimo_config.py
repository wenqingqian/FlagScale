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
    how many LLM microbatches one ViT macro forward serves.  vbf must be > 1;
    vbf == 1 is rejected by ``validate_mimo_config``.
    """
    vision_samples = vision_data_parallel_size * vision_micro_batch_size
    language_samples = language_data_parallel_size * language_micro_batch_size
    # Reject silent floor-division truncation (e.g. 16 // 12) that would
    # silently shrink the effective vit_batch_factor.
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


def validate_mimo_config(
    args,
    vision_parallelism: ModuleParallelismConfig,
    language_parallelism: ModuleParallelismConfig,
    num_microbatches: int,
) -> int:
    """Validate all model-agnostic MIMO configuration constraints in one place.

    Covers: DDP overlap flags (overlap_param_gather/overlap_grad_reduce),
    legacy-torch ckpt format, CP/PP == 1 for both modules, vision TP == 1,
    vit_batch_factor > 1, and vit_batch_factor % language_tp == 0; each
    assert message states its own reason.  Training entries must call this
    once before constructing the MIMO model; do not duplicate these checks
    elsewhere.

    Returns the validated ``vit_batch_factor``.
    """
    assert not (args.overlap_param_gather or args.overlap_grad_reduce), (
        "MIMO per-module DDP overlap is not yet validated; "
        "disable --overlap-param-gather and --overlap-grad-reduce."
    )
    assert args.ckpt_format == "torch", (
        "ChainedOptimizer returns a list of optimizer state dicts, which only the "
        "legacy 'torch' checkpoint path handles; torch_dist support is future work."
    )
    for name, parallelism in (("vision", vision_parallelism), ("language", language_parallelism)):
        assert (
            parallelism.context_parallel_size == 1 and parallelism.pipeline_model_parallel_size == 1
        ), (
            f"Colocated MIMO currently requires CP=1 and PP=1, got {name} "
            f"cp={parallelism.context_parallel_size}, "
            f"pp={parallelism.pipeline_model_parallel_size}."
        )
    assert vision_parallelism.tensor_model_parallel_size == 1, (
        f"Colocated MIMO currently requires vision TP=1, got "
        f"vision tp={vision_parallelism.tensor_model_parallel_size}."
    )

    vision_micro_batch_size = getattr(args, "vision_micro_batch_size", args.micro_batch_size)
    vit_batch_factor = compute_vit_batch_factor(
        vision_data_parallel_size=vision_parallelism.data_parallel_size,
        vision_micro_batch_size=vision_micro_batch_size,
        language_data_parallel_size=language_parallelism.data_parallel_size,
        language_micro_batch_size=args.micro_batch_size,
        num_microbatches=num_microbatches,
    )
    assert vit_batch_factor > 1, (
        f"MIMO requires vit_batch_factor > 1, got {vit_batch_factor} "
        f"(gbs={args.global_batch_size}, "
        f"language: tp={language_parallelism.tensor_model_parallel_size}, "
        f"dp={language_parallelism.data_parallel_size}, mbs={args.micro_batch_size}; "
        f"vision: tp={vision_parallelism.tensor_model_parallel_size}, "
        f"dp={vision_parallelism.data_parallel_size}, mbs={vision_micro_batch_size}). "
        f"The vbf=1 direct path duplicates ViT compute without macro batching; "
        f"consider disabling MIMO (use_mimo: false) for this configuration."
    )
    language_tp = language_parallelism.tensor_model_parallel_size
    assert vit_batch_factor % language_tp == 0, (
        f"vit_batch_factor ({vit_batch_factor}) must be a multiple of language TP "
        f"({language_tp}) for the intra-TP owner mapping. "
        f"Adjust vision_micro_batch_size or micro_batch_size."
    )
    return vit_batch_factor
