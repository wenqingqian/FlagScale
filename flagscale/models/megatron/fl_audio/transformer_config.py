"""Transformer config builders for FL Audio (TeleFLM + DepthGPT)."""

import torch
import torch.nn.functional as F
from copy import deepcopy

def _update_common_config(args, config):
    # muP
    config.use_mup = args.use_mup
    config.input_mult = args.input_mult
    config.output_mult = args.output_mult
    config.mup_scale_factor = args.mup_scale_factor

    # Audio-specific
    config.vocab_size = args.vocab_size
    config.num_channel = args.num_channel
    config.aud_emp_token_id = args.aud_emp_token_id
    config.loss_weights = args.loss_weights

    config.micro_batch_size = args.micro_batch_size
    config.seq_length = args.seq_length

    return config

def get_teleflm_config(args, config):
    """Build TransformerConfig for the TeleFLM (main) model from args and base config."""
    config = deepcopy(config)
    config = _update_common_config(args, config)

    # No PP for this sub-model
    config.pipeline_model_parallel_size = 1

    # Bias
    config.add_bias_linear = args.add_bias_linear
    config.add_qkv_bias = args.add_qkv_bias

    return config


def get_depth_gpt_config(args, config):
    """Build TransformerConfig for the DepthGPT (second) model from args and base config."""
    config = deepcopy(config)
    config = _update_common_config(args, config)

    # Architecture (from depthgpt extra-args)
    config.hidden_size = args.depth_hidden_size  # 1024
    config.num_attention_heads = args.depth_n_head  # 16
    config.num_query_groups = args.depth_n_head  # no GQA
    config.num_layers = args.depth_n_layer
    config.kv_channels = config.hidden_size // config.num_attention_heads

    # SwiGLU intermediate: int(8 * n_embd / 3)
    if args.swiglu:
        config.ffn_hidden_size = int(8 * args.depth_hidden_size / 3)
        config.gated_linear_unit = True
        config.activation_func = F.silu
    else:
        config.ffn_hidden_size = 4 * args.depth_hidden_size
        config.gated_linear_unit = False

    config.bias_activation_fusion = False

    # Bias
    config.add_bias_linear = args.depth_bias
    config.add_qkv_bias = args.depth_bias

    # main model hidden_size for linear_in
    config.hidden_size_flm = args.hidden_size  
    config.use_cmlp = args.depth_use_channel_mlp
    config.loss_weights = args.loss_weights

    return config