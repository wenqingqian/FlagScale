import torch
from typing import Optional

from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from flagscale.models.megatron.fl_audio.teleflm import TeleFLMModel
from flagscale.models.megatron.fl_audio.depth_gpt import DepthGPT, DepthGPT_Preprocessor, DepthGPT_Postprocessor
from flagscale.models.megatron.fl_audio.layer_spec import FLAudioBackend


class TeleFLMForCausalLM(MegatronModule):

    def __init__(
        self,
        teleflm_config: TransformerConfig,
        teleflm_layer_spec: ModuleSpec,
        depth_config: TransformerConfig,
        depth_layer_spec: ModuleSpec,
        pre_process: bool = True,
        post_process: bool = True,
        backend: FLAudioBackend = None
    ):
        super().__init__(config=teleflm_config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.teleflm_config = teleflm_config
        self.depth_config = depth_config

        if self.pre_process:
            self.teleflm_model = TeleFLMModel(teleflm_config, teleflm_layer_spec, backend=backend)
            self.depth_preprocessor = DepthGPT_Preprocessor(depth_config)

        self.depth_gpt = DepthGPT(depth_config, depth_layer_spec)

        if self.post_process:
            self.depth_postprocessor = DepthGPT_Postprocessor(depth_config, backend=backend)

    def set_input_tensor(self, input_tensor) -> None:
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for fl-audio'

        if self.pre_process:
            self.teleflm_model.set_input_tensor(input_tensor[0])
        else:
            self.depth_gpt.set_input_tensor(input_tensor[0])

    def forward(
        self,
        audio_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        hidden_states = None

        if self.pre_process:
            hidden_states = self.teleflm_model(
                audio_ids=audio_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = self.depth_preprocessor(audio_ids, hidden_states)

        hidden_states = self.depth_gpt(hidden_states)

        if self.post_process:
            loss, channel_losses = self.depth_postprocessor(hidden_states, audio_ids, labels)
            return loss, channel_losses

        return hidden_states