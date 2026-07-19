# Copyright (c) 2025, BAAI. All rights reserved.

"""Generic MIMO building blocks for colocated deployment."""

from .hetero_pg_utils import build_colocated_pg_collections
from .mimo_config import MIMOParallelismConfig, ModuleParallelismConfig
from .mimo_optimizer import (
    ChainedOptimizer,
    build_mimo_ddp_config,
    build_mimo_optimizer,
    get_mimo_ddp_wrappers,
    patch_mimo_model_chunk,
    set_mimo_force_all_reduce,
    setup_mimo_ddp,
    wrap_mimo_ddp,
)
from .mimo_scheduler import MIMOMicrobatchScheduler
from .mimo_utils import reshard_between_tp
from .parallel_state_ctx import switch_parallel_state

__all__ = [
    "MIMOParallelismConfig",
    "ModuleParallelismConfig",
    "build_colocated_pg_collections",
    "switch_parallel_state",
    "reshard_between_tp",
    "MIMOMicrobatchScheduler",
    "build_mimo_ddp_config",
    "wrap_mimo_ddp",
    "ChainedOptimizer",
    "build_mimo_optimizer",
    "setup_mimo_ddp",
    "set_mimo_force_all_reduce",
    "get_mimo_ddp_wrappers",
    "patch_mimo_model_chunk",
]
