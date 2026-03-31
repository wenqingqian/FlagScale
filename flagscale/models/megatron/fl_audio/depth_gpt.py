import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from torch.nn import CrossEntropyLoss
from flagscale.models.megatron.fl_audio.embedding import ChannelParallelEmbedding
from flagscale.models.megatron.fl_audio.layer_spec import FLAudioBackend

class DepthGPT_Preprocessor(MegatronModule):
    def __init__(self, config):
        super().__init__(config=config)
        self.config = config
        self.num_channel = config.num_channel
        self.hidden_size = config.hidden_size
        self.aud_emp_token_id = config.aud_emp_token_id
        self.embedding = ChannelParallelEmbedding(config, reduce_channel=False)
        self.position_embed = torch.nn.Embedding(self.num_channel, self.hidden_size)
        self.linear_in = torch.nn.Linear(config.hidden_size_flm, config.hidden_size * config.num_channel, bias=False)
        self.dropout = torch.nn.Dropout(config.hidden_dropout)

    def forward(self, audio_ids, hidden_states):
        # [b, s, c] -> [b, s-1, c-1] -> [s-1, b, c-1] -> [(s-1)b, c-1]
        audio_ids = audio_ids[:, 1:, :-1].transpose(0, 1).contiguous().reshape((-1, audio_ids.shape[-1] - 1))
        # [(s-1)b, c-1] -> [(s-1)b, c]
        audio_ids = F.pad(audio_ids, (1, 0), value=self.aud_emp_token_id)
        # [(s-1)b, c] -> [c, (s-1)b, hd] DepthGPT treats c as sequence, (s-1)b as batch
        audio_hidden_states = self.embedding(audio_ids)

        # [s, b, hm] -> [s-1, b, hm] -> [(s-1)b, hm]
        hidden_states = hidden_states[:-1, :, :].reshape((-1, hidden_states.shape[-1]))
        # [(s-1)b, hm] @ [hm, c*hd] -> [(s-1)b, c, hd] -> [c, (s-1)b, hd]
        hidden_states = self.linear_in(hidden_states).view(hidden_states.shape[0], self.num_channel, -1).transpose(0, 1)

        # [c, 1, hd]
        position_embedding = self.position_embed(
            torch.arange(0, self.num_channel, dtype=torch.long, device=audio_ids.device)
        ).unsqueeze(1)

        output = audio_hidden_states + position_embedding + hidden_states
        output = self.dropout(output)
        return output

class DepthGPT(GPTModel):
    def __init__(
        self,
        config: TransformerConfig,
        depth_layer_spec: ModuleSpec
    ):
        super().__init__(config=config, transformer_layer_spec=depth_layer_spec,
                         vocab_size=0, max_sequence_length=0,
                         pre_process=False, post_process=False)

    def set_input_tensor(self, tensor):
        assert 0, "todo"

    def forward(
        self,
        hidden_states
    ) -> torch.Tensor:

        if hidden_states is not None:
            super().set_input_tensor(hidden_states)
        hidden_states = super().forward(
            input_ids=None,
            decoder_input=None,
            position_ids=None,
            attention_mask=None,
        )

        return hidden_states

class DepthGPT_Postprocessor(MegatronModule):
    def __init__(self, config, backend):
        super().__init__(config=config)
        self.config = config
        self.num_channel = config.num_channel
        self.aud_emp_token_id = config.aud_emp_token_id
        self.loss_weights = config.loss_weights

        self.norm = backend.layernorm()(
            config=config,
            hidden_size=config.hidden_size,
            eps=config.layernorm_epsilon
        )

        # TODO[WQQ] channel parallel
        self.lm_heads = nn.ModuleList([nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_channel)])

    def forward(self, hidden_states, audio_ids, labels=None):
        # hidden_states: [c, bs-1, hd]
        hidden_states = self.norm(hidden_states)
        # [c, bs-1, hd] -> [c, bs-1, vocab]
        hidden_states = torch.stack([self.lm_heads[c](hidden_states[c, :, :]) for c in range(self.num_channel)])

        # [c, bs-1, vocab] -> [bs-1, c, vocab] -> [b, s-1, c, vocab]
        hidden_states = hidden_states.transpose(0, 1)
        batch_size = audio_ids.shape[0]
        hidden_states = hidden_states.reshape(
            (
                batch_size,
                -1,
                hidden_states.shape[-2],
                hidden_states.shape[-1]
            )
        )

        loss, channel_losses = None, None
        if labels is not None:
            loss, channel_losses = self.compute_loss(hidden_states, labels)
        else:
            loss, channel_losses = self.compute_loss(hidden_states, audio_ids[:, 1:, :])

        return (loss, channel_losses)

    def compute_loss(self, logits, labels):
        # output -> [b s-1 c v]
        batch_size, seq_length, num_channel, vocab_size = logits.shape
        loss_mask = (labels != self.aud_emp_token_id).float()
        logits = logits.reshape((-1, vocab_size)).contiguous()
        labels = labels.reshape(-1).contiguous()

        # TODO parallel
        loss_fct = CrossEntropyLoss(reduction='none')
        token_loss = loss_fct(logits.float(), labels)
        token_loss = token_loss.view(batch_size, seq_length, num_channel)

        def _get_loss(_mask, _loss):
            masked_losses = _loss.float() * _mask
            mask_sum = _mask.view(-1).sum()
            if mask_sum == 0:
                training_loss = torch.sum(masked_losses.view(-1)) / 1
            else:
                training_loss = torch.sum(masked_losses.view(-1)) / mask_sum
            return training_loss

        channel_losses = []
        for c in range(num_channel):
            channel_loss = _get_loss(loss_mask[:, :, c], token_loss[:, :, c])
            channel_losses.append(channel_loss)

        loss = None
        for _l, _w in zip(channel_losses, self.loss_weights):
            if loss is not None:
                loss += _l * _w
            else:
                loss = _l * _w
        return loss, channel_losses