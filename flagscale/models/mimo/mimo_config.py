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
