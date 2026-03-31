import torch

from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.torch_norm import WrappedTorchNorm

try:
    from megatron.core.transformer.custom_layers.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TENorm,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import apex

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn(f'Apex is not installed. Falling back to Torch LayerNorm')
    LNImpl = WrappedTorchNorm

class CMLP(MegatronModule):
    """Channel-wise MLP: each channel has an independent MLP.

    Replaces the standard MLP in DepthGPT transformer layers.
    Input shape: [seq, batch * num_channel, hidden] (Megatron uses seq-first).
    The channel dimension is folded into batch by the caller (DepthGPT).
    Each channel slice is routed to its own MLP.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        is_expert: bool = False,
        input_size=None,
    ):
        super().__init__(config=config)
        self.num_channel = config.num_channel
        self.mlps = torch.nn.ModuleList([
            MLP(config=config, submodules=submodules, is_expert=is_expert, input_size=input_size)
            for _ in range(self.num_channel)
        ])

    def forward(self, hidden_states, per_token_scale=None):
        """Forward pass.

        hidden_states: [seq_len, batch_size, num_channel * hidden_size] is NOT the case.
        Actually in DepthGPT the input is [num_channel, batch_size, hidden_size] where
        num_channel acts as the sequence dimension. Each channel gets its own MLP.
        """
        # hidden_states shape: [num_channel, batch_size, hidden_size]
        outputs = []
        biases = []
        for c in range(self.num_channel):
            out, bias = self.mlps[c](hidden_states[c:c+1, :, :], per_token_scale=per_token_scale)
            outputs.append(out)
            biases.append(bias)

        output = torch.cat(outputs, dim=0)
        # Aggregate bias: if any bias is not None, stack them; otherwise None
        if biases[0] is not None:
            assert 0, "DEBUG, bias should not exist"
            output_bias = torch.cat(biases, dim=0)
        else:
            output_bias = None

        return output, output_bias

class FLAudioBackend:
    def __init__(self, config):
        global HAVE_TE
        self.config = config
        self.use_te = config.use_te
        assert not (self.use_te and not HAVE_TE), f"use_te={self.use_te} while have_te={HAVE_TE}"
    def layernorm(self):
        if self.use_te and self.config.normalization == "RMSNorm":
            return TENorm
        return LNImpl
    def column_parallel_linear(self):
        return TEColumnParallelLinear if self.use_te else ColumnParallelLinear
    def row_parallel_linear(self):
        return TERowParallelLinear if self.use_te else RowParallelLinear
    def core_attention(self):
        return TEDotProductAttention if self.use_te else DotProductAttention

def get_attn_module_spec(backend):
    return ModuleSpec(
        module=SelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=SelfAttentionSubmodules(
            linear_qkv=backend.column_parallel_linear(),
            core_attention=backend.core_attention(),
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=IdentityOp,
            k_layernorm=IdentityOp,
        )
    )


def _get_mlp_submodules(backend):
    return MLPSubmodules(
        linear_fc1=backend.column_parallel_linear(),
        linear_fc2=backend.row_parallel_linear(),
    )


def get_teleflm_layer_spec(backend):
    """Layer spec for TeleFLM (main model). Standard MLP."""
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=backend.layernorm(),
            self_attention=get_attn_module_spec(backend),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=backend.layernorm(),
            mlp=ModuleSpec(
                module=MLP,
                submodules=_get_mlp_submodules(backend),
            ),
            mlp_bda=get_bias_dropout_add,
        )
    )


def get_depth_gpt_layer_spec(backend, use_cmlp=True):
    """Layer spec for DepthGPT. Uses CMLP when use_cmlp=True."""
    mlp_module = CMLP if use_cmlp else MLP
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=backend.layernorm(),
            self_attention=get_attn_module_spec(backend),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=backend.layernorm(),
            mlp=ModuleSpec(
                module=mlp_module,
                submodules=_get_mlp_submodules(backend),
            ),
            mlp_bda=get_bias_dropout_add,
        )
    )