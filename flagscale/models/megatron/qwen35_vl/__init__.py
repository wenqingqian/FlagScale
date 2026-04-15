# Copyright (c) 2025, BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

from .model import Qwen35VLModel
from .language_model import Qwen35VLLanguageModule
from .transformer_config import Qwen35VLTransformerConfig, get_vision_model_config, get_vision_projection_config
from .layer_specs import get_qwen35vl_language_model_spec
from .attention import Qwen35VLSelfAttention
from .rope import Qwen35VLLanguageRotaryEmbedding, get_rope_index

# Re-export vision model spec from qwen3_vl (identical vision encoder)
from flagscale.models.megatron.qwen3_vl.layer_specs import get_qwen3vl_vision_model_spec

__all__ = [
    "Qwen35VLModel",
    "Qwen35VLLanguageModule",
    "Qwen35VLTransformerConfig",
    "get_vision_model_config",
    "get_vision_projection_config",
    "get_qwen35vl_language_model_spec",
    "get_qwen3vl_vision_model_spec",
    "Qwen35VLSelfAttention",
    "Qwen35VLLanguageRotaryEmbedding",
    "get_rope_index",
]
