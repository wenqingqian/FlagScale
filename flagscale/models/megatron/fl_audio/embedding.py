import math
import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel.mappings import (
    reduce_from_tensor_model_parallel_region,
)

class ChannelParallelEmbedding(torch.nn.Module):
    """Hybrid channel-vocab parallel embedding.

    Given vocab_parallel_size (vp), derives cp = tp_size / vp.
    Each rank holds channels_per_tp = num_channel / cp embeddings,
    each with vocab_per_tp = padded_vocab_size / vp rows.

    Default vp=1 means pure channel parallel (cp=tp_size).
    A single all-reduce combines both channel partial sums and vocab partial sums.
    """

    def __init__(self, config, reduce_channel=None):
        super().__init__()
        self.config = config
        self.use_mup = config.use_mup
        self.input_mult = config.input_mult
        self.hidden_size = config.hidden_size
        self.num_channel = config.num_channel
        self.reduce_channel = reduce_channel

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        vp = getattr(config, 'vocab_parallel_size', 1)

        assert tp_size % vp == 0, f"tp_size ({tp_size}) must be divisible by vp ({vp})"
        cp = tp_size // vp
        assert self.num_channel % cp == 0, (
            f"num_channel ({self.num_channel}) must be divisible by cp ({cp})"
        )

        if vp > 1:
            self.vocab_size = config.padded_vocab_size
        else:
            self.vocab_size = config.vocab_size

        self.channels_per_tp = self.num_channel // cp
        self.vocab_per_tp = self.vocab_size // vp if vp > 1 else self.vocab_size
        self.vp = vp
        self.vp_rank = tp_rank % vp
        self.cp_rank = tp_rank // vp
        self.channel_start = self.cp_rank * self.channels_per_tp
        self.vocab_start = self.vp_rank * self.vocab_per_tp

        self.audio_embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(self.vocab_per_tp, self.hidden_size)
            for _ in range(self.channels_per_tp)
        ])

    def _vocab_parallel_lookup(self, input_ids, embedding):
        """Lookup with vocab partitioning: out-of-range tokens produce zero vectors."""
        vocab_end = self.vocab_start + self.vocab_per_tp
        in_range = (input_ids >= self.vocab_start) & (input_ids < vocab_end)
        local_ids = (input_ids - self.vocab_start).clamp(min=0, max=self.vocab_per_tp - 1)
        output = embedding(local_ids)
        output = output * in_range.unsqueeze(-1).to(output.dtype)
        return output

    def _lookup(self, input_ids, embedding):
        """Dispatch to vocab-parallel or direct lookup based on vp."""
        if self.vp > 1:
            return self._vocab_parallel_lookup(input_ids, embedding)
        return embedding(input_ids)

    def forward(self, audio_ids):
        tp_size = get_tensor_model_parallel_world_size()

        # TeleFLM path: [b, s, c] -> [s, b, hm], sum over channels
        if self.reduce_channel:
            if audio_ids.shape[0] == self.config.micro_batch_size and \
                audio_ids.shape[1] == self.config.seq_length:
                audio_ids = audio_ids.transpose(0, 1).contiguous()

            embeddings = None
            for local_idx in range(self.channels_per_tp):
                global_idx = self.channel_start + local_idx
                emb = self._lookup(
                    audio_ids[..., global_idx], self.audio_embeddings[local_idx]
                )
                embeddings = emb if embeddings is None else embeddings + emb

            if tp_size > 1:
                embeddings = reduce_from_tensor_model_parallel_region(embeddings)
            if self.use_mup:
                embeddings = embeddings * self.input_mult
            return embeddings

        # DepthGPT path: [(s-1)b, c] -> [c, (s-1)b, hd]
        else:
            full = torch.zeros(
                self.num_channel, audio_ids.shape[0], self.hidden_size,
                dtype=self.audio_embeddings[0].weight.dtype,
                device=audio_ids.device,
            )
            for local_idx in range(self.channels_per_tp):
                global_idx = self.channel_start + local_idx
                full[global_idx] = self._lookup(
                    audio_ids[:, global_idx], self.audio_embeddings[local_idx]
                )

            if tp_size > 1:
                full = reduce_from_tensor_model_parallel_region(full)
            return full