"""Convert Megatron (FlagScale) checkpoint to HuggingFace (NativeAudio) format.

Handles TP merging, fused weight splitting, key remapping.
Uses distrib_optim.pt for fp32 master weights and optimizer states.
Produces a full NativeAudio checkpoint directory (model + optimizer + scheduler + scaler + rng + config).

Usage:
    python convert_mcore_to_hf.py \
        --mcore-dir /path/to/checkpoints \
        --output-dir /path/to/output \
        --tp-size 1

TODO:
    1. Only support tp.
    2. ! If not use mixed precision, load weight from model_optim_rng.pt.
"""

import argparse
import os
import sys
import json
import torch
from collections import OrderedDict
from safetensors.torch import save_file


# ============================================================================
# Config extraction from mcore args
# ============================================================================

def extract_model_config(mcore_args):
    """Extract required model/optimizer config from mcore checkpoint args.

    Raises ValueError if any required arg is missing.
    """
    required = {
        # teleflm model
        "num_attention_heads": "teleflm num_attention_heads",
        "num_query_groups": "teleflm num_query_groups",
        "hidden_size": "teleflm hidden_size",
        "num_channel": "audio num_channel",
        # depth model
        "depth_n_head": "depth num_heads",
        "depth_hidden_size": "depth hidden_size",
        # optimizer
        "lr": "learning rate",
        "weight_decay": "weight decay",
        "adam_beta1": "adam beta1",
        "adam_beta2": "adam beta2",
        "adam_eps": "adam epsilon",
    }
    config = {}
    missing = []
    for key, desc in required.items():
        val = getattr(mcore_args, key, None)
        if val is None:
            missing.append(f"  {key} ({desc})")
        else:
            config[key] = val
    if missing:
        raise ValueError(
            "Missing required args in mcore checkpoint:\n" + "\n".join(missing)
        )

    # Derived params
    config["teleflm_head_dim"] = config["hidden_size"] // config["num_attention_heads"]
    config["teleflm_heads_per_group"] = config["num_attention_heads"] // config["num_query_groups"]
    config["depth_head_dim"] = config["depth_hidden_size"] // config["depth_n_head"]

    return config

# ============================================================================
# TP merge strategy table
# "cat_dim": which dim to concatenate across TP ranks
#   - 0 = row parallel (output dim split)
#   - 1 = column parallel (input dim split)
#   - None = not TP-split (replicated), just take rank 0
#   - vocab_split = hybrid
# "split_type": how to post-process after merge
#   - None = use as-is
#   - "qkv_teleflm" = split fused QKV for teleflm (GQA: 8 heads, 4 kv heads)
#   - "qkv_depth" = split fused QKV for depth (MHA: 16 heads)
#   - "swiglu" = split fused gate+up projection
# ============================================================================

TP_STRATEGY = {
    "teleflm.layernorm":            {"cat_dim": None},
    "teleflm.linear_qkv.weight":    {"cat_dim": 0, "split_type": "qkv_teleflm"},
    "teleflm.linear_qkv.bias":      {"cat_dim": 0, "split_type": "qkv_bias_teleflm"},
    "teleflm.linear_proj.weight":   {"cat_dim": 1},
    "teleflm.linear_fc1.weight":    {"cat_dim": 0, "split_type": "swiglu"},
    "teleflm.linear_fc2.weight":    {"cat_dim": 1},
    "teleflm.embedding":            {"cat_dim": "vocab_split"},
    "teleflm.norm":                 {"cat_dim": None},
    "depth.layernorm":              {"cat_dim": None},
    "depth.linear_qkv.weight":      {"cat_dim": 0, "split_type": "qkv_depth"},
    "depth.linear_proj.weight":     {"cat_dim": 1},
    "depth.mlps.linear_fc1.weight": {"cat_dim": 0, "split_type": "swiglu"},
    "depth.mlps.linear_fc2.weight": {"cat_dim": 1},
    "depth.embedding":              {"cat_dim": "vocab_split"},
    "depth.lm_heads":               {"cat_dim": None},
    "depth.norm":                   {"cat_dim": None},
    "depth.linear_in":              {"cat_dim": None},
    "depth.position_embed":         {"cat_dim": None},
}


def classify_key(mcore_key):
    """Classify a megatron key to determine its TP merge strategy."""
    if mcore_key.endswith("_extra_state"):
        return None

    if mcore_key.startswith("teleflm_model."):
        if "embedding.audio_embeddings" in mcore_key:
            return "teleflm.embedding"
        if "norm.weight" in mcore_key and "layernorm" not in mcore_key and "decoder" not in mcore_key:
            return "teleflm.norm"
        if "input_layernorm" in mcore_key or "pre_mlp_layernorm" in mcore_key:
            return "teleflm.layernorm"
        if "linear_qkv.weight" in mcore_key:
            return "teleflm.linear_qkv.weight"
        if "linear_qkv.bias" in mcore_key:
            return "teleflm.linear_qkv.bias"
        if "linear_proj.weight" in mcore_key:
            return "teleflm.linear_proj.weight"
        if "linear_fc1.weight" in mcore_key:
            return "teleflm.linear_fc1.weight"
        if "linear_fc2.weight" in mcore_key:
            return "teleflm.linear_fc2.weight"

    elif mcore_key.startswith("depth_gpt."):
        if "input_layernorm" in mcore_key or "pre_mlp_layernorm" in mcore_key:
            return "depth.layernorm"
        if "linear_qkv.weight" in mcore_key:
            return "depth.linear_qkv.weight"
        if "linear_proj.weight" in mcore_key:
            return "depth.linear_proj.weight"
        if "mlps" in mcore_key and "linear_fc1" in mcore_key:
            return "depth.mlps.linear_fc1.weight"
        if "mlps" in mcore_key and "linear_fc2" in mcore_key:
            return "depth.mlps.linear_fc2.weight"

    elif mcore_key.startswith("depth_preprocessor."):
        if "audio_embeddings" in mcore_key:
            return "depth.embedding"
        if "linear_in" in mcore_key:
            return "depth.linear_in"
        if "position_embed" in mcore_key:
            return "depth.position_embed"

    elif mcore_key.startswith("depth_postprocessor."):
        if "lm_heads" in mcore_key:
            return "depth.lm_heads"
        if "norm" in mcore_key:
            return "depth.norm"

    return None


# ============================================================================
# TP merge / split helpers
# ============================================================================

def merge_tp(tensors, cat_dim):
    if cat_dim is None:
        return tensors[0]
    return torch.cat(tensors, dim=cat_dim)


def split_qkv_teleflm(merged, num_query_groups, heads_per_group, head_dim):
    """Split fused QKV for TeleFLM GQA."""
    per_group = (heads_per_group + 2) * head_dim

    q_chunks, k_chunks, v_chunks = [], [], []
    for g in range(num_query_groups):
        offset = g * per_group
        q_chunks.append(merged[offset:offset + heads_per_group * head_dim])
        q_end = offset + heads_per_group * head_dim
        k_chunks.append(merged[q_end:q_end + head_dim])
        v_chunks.append(merged[q_end + head_dim:q_end + 2 * head_dim])

    return torch.cat(q_chunks, dim=0), torch.cat(k_chunks, dim=0), torch.cat(v_chunks, dim=0)


def split_qkv_teleflm_bias(merged, num_query_groups, heads_per_group, head_dim):
    """Split fused QKV bias for TeleFLM GQA. Same interleave as weight dim 0."""
    return split_qkv_teleflm(merged, num_query_groups, heads_per_group, head_dim)


def split_qkv_depth(merged, num_heads, head_dim):
    """Split fused QKV for Depth MHA."""
    per_head = 3 * head_dim

    q_chunks, k_chunks, v_chunks = [], [], []
    for h in range(num_heads):
        offset = h * per_head
        q_chunks.append(merged[offset:offset + head_dim])
        k_chunks.append(merged[offset + head_dim:offset + 2 * head_dim])
        v_chunks.append(merged[offset + 2 * head_dim:offset + per_head])

    return torch.cat(q_chunks, dim=0), torch.cat(k_chunks, dim=0), torch.cat(v_chunks, dim=0)


def split_swiglu(merged):
    """Split fused SwiGLU: [gate_proj; up_proj] along dim 0."""
    half = merged.shape[0] // 2
    return merged[:half], merged[half:]


# ============================================================================
# Key remapping
# ============================================================================

def _extract_layer_num(key, prefix):
    rest = key[len(prefix):]
    return int(rest.split(".")[0])


def _extract_mlp_idx(key):
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "mlps":
            return int(parts[i + 1])
    raise ValueError(f"Cannot extract mlp index from {key}")


def _remap_embedding(mcore_key, rank, tp_size, num_channel):
    parts = mcore_key.split(".")
    for i, p in enumerate(parts):
        if p == "audio_embeddings":
            local_idx = int(parts[i + 1])
            break
    else:
        return None

    channels_per_rank = num_channel // tp_size
    global_idx = rank * channels_per_rank + local_idx

    if "teleflm_model" in mcore_key:
        return f"model.embed_tokens.audio_embeddings.{global_idx}.weight"
    elif "depth_preprocessor" in mcore_key:
        return f"aud_output_layers.transformer.wtes.{global_idx}.weight"
    return None


def remap_key(mcore_key):
    """Remap a single megatron key to HF key (for non-split weights)."""
    if mcore_key.startswith("teleflm_model.decoder.layers."):
        layer = _extract_layer_num(mcore_key, "teleflm_model.decoder.layers.")
        suffix = mcore_key.split(f"layers.{layer}.")[-1]
        mapping = {
            "input_layernorm.weight": f"model.layers.{layer}.input_layernorm.weight",
            "pre_mlp_layernorm.weight": f"model.layers.{layer}.post_attention_layernorm.weight",
            "self_attention.linear_proj.weight": f"model.layers.{layer}.self_attn.o_proj.weight",
            "mlp.linear_fc2.weight": f"model.layers.{layer}.mlp.down_proj.weight",
        }
        return mapping.get(suffix)

    elif mcore_key == "teleflm_model.norm.weight":
        return "model.norm.weight"

    elif mcore_key.startswith("depth_gpt.decoder.layers."):
        layer = _extract_layer_num(mcore_key, "depth_gpt.decoder.layers.")
        suffix = mcore_key.split(f"layers.{layer}.")[-1]
        if suffix == "input_layernorm.weight":
            return f"aud_output_layers.transformer.h.{layer}.ln_1.weight"
        elif suffix == "pre_mlp_layernorm.weight":
            return f"aud_output_layers.transformer.h.{layer}.ln_2.weight"
        elif suffix == "self_attention.linear_proj.weight":
            return f"aud_output_layers.transformer.h.{layer}.attn.c_proj.weight"
        elif "mlps" in suffix and "linear_fc2" in suffix:
            mlp_idx = _extract_mlp_idx(mcore_key)
            return f"aud_output_layers.transformer.h.{layer}.mlps.{mlp_idx}.down_proj.weight"

    elif mcore_key == "depth_preprocessor.linear_in.weight":
        return "aud_output_layers.linear_in.weight"
    elif mcore_key == "depth_preprocessor.position_embed.weight":
        return "aud_output_layers.transformer.wpe.weight"

    elif mcore_key.startswith("depth_postprocessor.lm_heads."):
        idx = mcore_key.split("lm_heads.")[1].split(".")[0]
        return f"aud_output_layers.lm_heads.{idx}.weight"
    elif mcore_key == "depth_postprocessor.norm.weight":
        return "aud_output_layers.transformer.ln_f.weight"

    return None


# ============================================================================
# Unflatten distrib_optim tensors
# ============================================================================

def unflatten_distrib_optim(flat_param, flat_exp_avg, flat_exp_avg_sq, model_keys, model_state):
    """Slice flat optimizer tensors back into per-key dicts matching model_state shapes.

    Args:
        flat_param: 1D fp32 tensor (master weights)
        flat_exp_avg: 1D fp32 tensor (Adam first moment)
        flat_exp_avg_sq: 1D fp32 tensor (Adam second moment)
        model_keys: ordered list of model keys (excluding _extra_state)
        model_state: dict of model tensors (for shape reference)

    Returns:
        Three dicts: {mcore_key: tensor} for param, exp_avg, exp_avg_sq
    """
    param_dict = OrderedDict()
    avg_dict = OrderedDict()
    avg_sq_dict = OrderedDict()
    # offset = 0

    flat_param_numel = flat_param.numel()

    for k in model_keys:
        shape = model_state[k].shape
        numel = model_state[k].numel()
        # NOTE
        # It seems that the data is stored in the reverse order of model_optim_rng and distrib_optim
        # So if use masters weight in distrib_optim.pt instead of low precision weights in model_optim_rng.pt
        #   data should be taken from back to front
        # See ./tools/inspect_distrib_optim.py
        param_dict[k] = flat_param[flat_param_numel-numel:flat_param_numel].reshape(shape)
        
        # TODO[WQQ] So do the following two also need to be taken in reverse order?
        # How to verify?
        # avg_dict[k] = flat_exp_avg[offset:offset + numel].reshape(shape)
        # avg_sq_dict[k] = flat_exp_avg_sq[offset:offset + numel].reshape(shape)
        avg_dict[k] = flat_exp_avg[flat_param_numel-numel:flat_param_numel].reshape(shape)
        avg_sq_dict[k] = flat_exp_avg_sq[flat_param_numel-numel:flat_param_numel].reshape(shape)
        flat_param_numel -= numel
        # offset += numel
    return param_dict, avg_dict, avg_sq_dict

# ============================================================================
# Core: convert a set of per-key tensors (works for param, exp_avg, exp_avg_sq)
# ============================================================================

def convert_tensor_dict(rank_dicts, tp_size, model_config, to_fp32=True):
    """Convert a list of per-rank {mcore_key: tensor} dicts to HF format.

    Applies the same TP merge + split + remap logic as model weight conversion.

    Args:
        rank_dicts: list of dicts, one per TP rank
        tp_size: tensor parallelism size
        to_fp32: whether to cast to fp32

    Returns:
        OrderedDict of {hf_key: tensor}
    """
    all_keys = sorted(rank_dicts[0].keys())
    hf_out = OrderedDict()

    for mcore_key in all_keys:
        category = classify_key(mcore_key)
        if category is None:
            continue

        strategy = TP_STRATEGY.get(category)
        if strategy is None:
            continue

        cat_dim = strategy["cat_dim"]
        split_type = strategy.get("split_type")

        rank_tensors = [rank_dicts[r][mcore_key] for r in range(tp_size)]

        # Vocab-split embeddings
        if cat_dim == "vocab_split":
            for r in range(tp_size):
                t = rank_tensors[r].float() if to_fp32 else rank_tensors[r]
                hf_key = _remap_embedding(mcore_key, r, tp_size, model_config["num_channel"])
                if hf_key:
                    hf_out[hf_key] = t
            continue

        merged = merge_tp(rank_tensors, cat_dim)
        if to_fp32:
            merged = merged.float()

        # QKV splits
        if split_type == "qkv_teleflm":
            q, k, v = split_qkv_teleflm(
                merged, model_config["num_query_groups"],
                model_config["teleflm_heads_per_group"], model_config["teleflm_head_dim"])
            layer = _extract_layer_num(mcore_key, "teleflm_model.decoder.layers.")
            hf_out[f"model.layers.{layer}.self_attn.q_proj.weight"] = q
            hf_out[f"model.layers.{layer}.self_attn.k_proj.weight"] = k
            hf_out[f"model.layers.{layer}.self_attn.v_proj.weight"] = v
            continue

        elif split_type == "qkv_bias_teleflm":
            q, k, v = split_qkv_teleflm_bias(
                merged, model_config["num_query_groups"],
                model_config["teleflm_heads_per_group"], model_config["teleflm_head_dim"])
            layer = _extract_layer_num(mcore_key, "teleflm_model.decoder.layers.")
            hf_out[f"model.layers.{layer}.self_attn.q_proj.bias"] = q
            hf_out[f"model.layers.{layer}.self_attn.k_proj.bias"] = k
            hf_out[f"model.layers.{layer}.self_attn.v_proj.bias"] = v
            continue

        elif split_type == "qkv_depth":
            q, k, v = split_qkv_depth(
                merged, model_config["depth_n_head"], model_config["depth_head_dim"])
            layer = _extract_layer_num(mcore_key, "depth_gpt.decoder.layers.")
            c_attn = torch.cat([q, k, v], dim=0)
            hf_out[f"aud_output_layers.transformer.h.{layer}.attn.c_attn.weight"] = c_attn
            continue

        elif split_type == "swiglu":
            gate, up = split_swiglu(merged)
            if "teleflm_model" in mcore_key:
                layer = _extract_layer_num(mcore_key, "teleflm_model.decoder.layers.")
                hf_out[f"model.layers.{layer}.mlp.gate_proj.weight"] = gate
                hf_out[f"model.layers.{layer}.mlp.up_proj.weight"] = up
            elif "depth_gpt" in mcore_key:
                layer = _extract_layer_num(mcore_key, "depth_gpt.decoder.layers.")
                mlp_idx = _extract_mlp_idx(mcore_key)
                hf_out[f"aud_output_layers.transformer.h.{layer}.mlps.{mlp_idx}.gate_proj.weight"] = gate
                hf_out[f"aud_output_layers.transformer.h.{layer}.mlps.{mlp_idx}.up_proj.weight"] = up
            continue

        # Direct remap
        hf_key = remap_key(mcore_key)
        if hf_key:
            hf_out[hf_key] = merged

    return hf_out


# ============================================================================
# Main conversion
# ============================================================================

def convert(mcore_dir, output_dir, tp_size, iteration=None,
            use_param_order_cache=True, hf_model_path=None, hf_config_path=None):
    # Resolve iteration
    if iteration is None:
        with open(os.path.join(mcore_dir, "latest_checkpointed_iteration.txt")) as f:
            iteration = int(f.read().strip())
    print(f"Converting iteration {iteration}, TP={tp_size}")

    iter_dir = os.path.join(mcore_dir, f"iter_{iteration:07d}")

    # Load model state dicts (for key order and shapes)
    model_shards = []
    for r in range(tp_size):
        rank_dir = os.path.join(iter_dir, f"mp_rank_{r:02d}")
        ckpt = torch.load(
            os.path.join(rank_dir, "model_optim_rng.pt"),
            map_location="cpu", weights_only=False
        )
        model_shards.append(ckpt)
    print("Loaded model_optim_rng.pt for all TP ranks")

    # Extract model config from mcore args
    mcore_args = model_shards[0]["args"]
    model_config = extract_model_config(mcore_args)
    print(f"Model config: teleflm heads={model_config['num_attention_heads']}, "
          f"kv_groups={model_config['num_query_groups']}, "
          f"depth heads={model_config['depth_n_head']}, "
          f"channels={model_config['num_channel']}")

    # Load distrib_optim.pt for fp32 master weights + optimizer states
    distrib_shards = []
    for r in range(tp_size):
        rank_dir = os.path.join(iter_dir, f"mp_rank_{r:02d}")
        dopt = torch.load(
            os.path.join(rank_dir, "distrib_optim.pt"),
            map_location="cpu", weights_only=False
        )
        inner = dopt[0][(torch.bfloat16, torch.float32)]
        distrib_shards.append(inner)
    print("Loaded distrib_optim.pt for all TP ranks")

    # Unflatten distrib_optim into per-key dicts for each rank
    param_dicts = []
    avg_dicts = []
    avg_sq_dicts = []
    for r in range(tp_size):
        model_state = model_shards[r]["model"]
        model_keys = [k for k in model_state.keys() if not k.endswith("_extra_state")]
        p, a, asq = unflatten_distrib_optim(
            distrib_shards[r]["param"],
            distrib_shards[r]["exp_avg"],
            distrib_shards[r]["exp_avg_sq"],
            model_keys, model_state
        )
        param_dicts.append(p)
        avg_dicts.append(a)
        avg_sq_dicts.append(asq)
    print("Unflattened optimizer states")

    # If use master weight from distrib_optim.pt, 
    # Convert model weights (from fp32 master weights)
    print("\n--- Converting model weights ---")
    hf_model = convert_tensor_dict(param_dicts, tp_size, model_config, to_fp32=True)
    print(f"Model: {len(hf_model)} keys")

    # Convert exp_avg
    print("--- Converting exp_avg ---")
    hf_exp_avg = convert_tensor_dict(avg_dicts, tp_size, model_config, to_fp32=True)
    print(f"exp_avg: {len(hf_exp_avg)} keys")

    # Convert exp_avg_sq
    print("--- Converting exp_avg_sq ---")
    hf_exp_avg_sq = convert_tensor_dict(avg_sq_dicts, tp_size, model_config, to_fp32=True)
    print(f"exp_avg_sq: {len(hf_exp_avg_sq)} keys")

    # Verify all three have same keys
    assert set(hf_model.keys()) == set(hf_exp_avg.keys()) == set(hf_exp_avg_sq.keys()), \
        "Key mismatch between model/exp_avg/exp_avg_sq"

    # Get HF model parameter order (must match optimizer param index)
    # Cache lives next to this script as param_order_cache.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(script_dir, "param_order_cache.json")

    if use_param_order_cache and os.path.exists(cache_path):
        with open(cache_path) as f:
            param_order = json.load(f)
        print(f"Loaded param_order from cache: {cache_path} ({len(param_order)} params)")
    else:
        if not hf_model_path or not hf_config_path:
            raise ValueError(
                "No param_order cache found. Provide --hf-model-path and --hf-config-path "
                "to generate it."
            )
        sys.path.insert(0, hf_model_path)
        from src.models import get_model_class
        with open(hf_config_path) as f:
            hf_config_dict = json.load(f)
        ConfigClass, ModelClass = get_model_class("audio_model")
        hf_config = ConfigClass(**hf_config_dict)
        hf_model_ref = ModelClass(hf_config)
        param_order = [name for name, _ in hf_model_ref.named_parameters()]
        del hf_model_ref
        # Save cache for next time
        with open(cache_path, "w") as f:
            json.dump(param_order, f, indent=2)
        print(f"Saved param_order cache to {cache_path}")
        print(f"HF model parameter order: {len(param_order)} params")

    # Verify all converted keys are in param_order
    missing = set(hf_model.keys()) - set(param_order)
    extra = set(param_order) - set(hf_model.keys())
    if missing:
        print(f"WARNING: converted keys not in model params: {missing}")
    if extra:
        print(f"WARNING: model params not in converted keys: {extra}")

    # Build NativeAudio optimizer state_dict
    # Keys are indexed 0..N-1 in the order of model.named_parameters()
    mcore_step = model_shards[0]["iteration"]
    opt_state = {}
    for i, param_name in enumerate(param_order):
        opt_state[i] = {
            "step": torch.tensor(float(mcore_step), dtype=torch.float32),
            "exp_avg": hf_exp_avg[param_name],
            "exp_avg_sq": hf_exp_avg_sq[param_name],
        }
    hf_keys = param_order


    # Build param_groups from model_config (already validated)
    lr = model_config["lr"]
    weight_decay = model_config["weight_decay"]
    adam_beta1 = model_config["adam_beta1"]
    adam_beta2 = model_config["adam_beta2"]
    adam_eps = model_config["adam_eps"]

    optimizer_state_dict = {
        "state": opt_state,
        "param_groups": [{
            "lr": lr,
            "betas": (adam_beta1, adam_beta2),
            "eps": adam_eps,
            "weight_decay": weight_decay,
            "amsgrad": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
            "decoupled_weight_decay": True,
            "initial_lr": lr,
            "params": list(range(len(hf_keys))),
        }],
    }

    # Build scheduler state_dict (LambdaLR-compatible)
    mcore_sched = model_shards[0]["opt_param_scheduler"]
    scheduler_state_dict = {
        "base_lrs": [lr],
        "last_epoch": mcore_step,
        "_step_count": mcore_step + 1,
        "_is_initial": False,
        "_get_lr_called_within_step": False,
        "_last_lr": [lr],  # will be overridden by scheduler on first step
        "lr_lambdas": [None],
    }

    # Build scaler state_dict.
    # Megatron bf16 doesn't use PyTorch GradScaler, so no scaler state in mcore ckpt.
    # bf16 has same dynamic range as fp32, so scale stays at init value (no underflow).
    # Values below are torch.amp.GradScaler defaults.
    scaler_state_dict = {
        "scale": 65536.0,           # GradScaler default init_scale
        "growth_factor": 2.0,       # GradScaler default
        "backoff_factor": 0.5,      # GradScaler default
        "growth_interval": 2000,    # GradScaler default
        "_growth_tracker": mcore_step,
    }

    # Build rng state (convert Megatron format to NativeAudio format)
    rng_state_list = model_shards[0]["rng_state"]
    if len(rng_state_list) == 1:
        print("WARNING: only 1 rng_state, need data_parallel_random_init=True for multi-DP")
    native_rng_state = {
        "cpu": rng_state_list[0]["torch_rng_state"],
        "numpy": rng_state_list[0]["np_rng_state"],
        "cuda": [r["cuda_rng_state"] for r in rng_state_list],
    }

    # Build dataloader state.
    # Native saves step+1 (1-indexed), so config["step"]=N means steps 0..N-1 done.
    # resume: start_step = config["step"] + 1, but native loop is 0-indexed,
    # so we store mcore_step - 1 to get start_step = mcore_step (run 1100 steps).
    native_step = mcore_step - 1
    dataloader_state = {"step": native_step}

    # Build config
    config = {"step": native_step, "is_ddp": True}

    # Save everything
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- Saving to {output_dir} ---")

    # model.safetensors
    save_file(hf_model, os.path.join(output_dir, "model.safetensors"))
    print("Saved model.safetensors")

    # optimizer.pt
    torch.save(optimizer_state_dict, os.path.join(output_dir, "optimizer.pt"))
    print("Saved optimizer.pt")

    # scheduler.pt
    torch.save(scheduler_state_dict, os.path.join(output_dir, "scheduler.pt"))
    print("Saved scheduler.pt")

    # scaler.pt
    torch.save(scaler_state_dict, os.path.join(output_dir, "scaler.pt"))
    print("Saved scaler.pt")

    # rng_state.pt
    torch.save(native_rng_state, os.path.join(output_dir, "rng_state.pt"))
    print("Saved rng_state.pt")

    # dataloader_state.pt
    torch.save(dataloader_state, os.path.join(output_dir, "dataloader_state.pt"))
    print("Saved dataloader_state.pt")

    # config.json
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print("Saved config.json")

    # Dump key list
    with open(os.path.join(output_dir, "converted_keys.txt"), "w") as f:
        for k in hf_keys:
            t = hf_model[k]
            f.write(f"{k}: {list(t.shape)} {t.dtype}\n")
    print("Saved converted_keys.txt")

    print(f"\nDone! Full checkpoint at {output_dir} (step={mcore_step})")


def main():
    parser = argparse.ArgumentParser(description="Convert Megatron ckpt to HF format")
    parser.add_argument("--mcore-dir", type=str, required=True,
                        help="Path to megatron checkpoints dir")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to output HF checkpoint dir")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--iteration", type=int, default=None,
                        help="Iteration to convert (default: latest)")
    parser.add_argument("--no-param-order-cache", dest="use_param_order_cache",
                        action="store_false", default=True,
                        help="Force regenerate param_order from HF model instead of using cache")
    parser.add_argument("--hf-model-path", type=str, default=None,
                        help="Path to NativeAudio project root (for param order generation)")
    parser.add_argument("--hf-config-path", type=str, default=None,
                        help="Path to HF model config JSON")
    args = parser.parse_args()

    convert(args.mcore_dir, args.output_dir, args.tp_size, args.iteration,
            args.use_param_order_cache, args.hf_model_path, args.hf_config_path)


if __name__ == "__main__":
    main()
