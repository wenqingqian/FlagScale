import torch
from typing import Optional

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from flagscale.models.megatron.fl_audio.layer_spec import FLAudioBackend
from flagscale.models.megatron.fl_audio.embedding import ChannelParallelEmbedding

class TeleFLMModel(GPTModel):
    def __init__(
        self,
        config: TransformerConfig,
        teleflm_layer_spec: ModuleSpec,
        backend: FLAudioBackend
    ):
        super().__init__(config=config, transformer_layer_spec=teleflm_layer_spec,
                         vocab_size=0, max_sequence_length=0,
                         pre_process=False, post_process=False,
                         rotary_base=config.rotary_base)

        self.norm = backend.layernorm()(
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon
        )
        self.embedding = ChannelParallelEmbedding(config, reduce_channel=True)

    def set_input_tensor(self, tensor):
        pass

    def forward(
        self,
        audio_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embedding(audio_ids)

        super().set_input_tensor(hidden_states)
        hidden_states = super().forward(input_ids=None, decoder_input=None,
                        position_ids=position_ids, attention_mask=attention_mask)

        hidden_states = self.norm(hidden_states)
        return hidden_states