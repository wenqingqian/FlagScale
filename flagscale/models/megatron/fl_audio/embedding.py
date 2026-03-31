import torch

class ChannelParallelEmbedding(torch.nn.Module):

    # TODO[WQQ] parallel
    def __init__(self, config, reduce_channel=None):
        super().__init__()
        self.config = config
        self.use_mup = config.use_mup
        self.input_mult = config.input_mult
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_channel = config.num_channel
        self.reduce_channel = reduce_channel
        self.audio_embeddings = torch.nn.ModuleList(
            [
                torch.nn.Embedding(self.vocab_size, self.hidden_size)
                for _ in range(self.num_channel)
            ]
        )

    def forward(self, audio_ids):
        # for teleflm [b, s, c] -> [s, b, hm] # transpose for standard gpt input
        if self.reduce_channel:
            if audio_ids.shape[0] == self.config.micro_batch_size and\
                audio_ids.shape[1] == self.config.seq_length:
                audio_ids = audio_ids.transpose(0, 1).contiguous()
            # [s, b, c]
            embeddings = None
            for aud_chn_idx in range(self.num_channel):
                aud_speak_embed = self.audio_embeddings[aud_chn_idx](audio_ids[..., aud_chn_idx])
                if embeddings is None:
                    embeddings = aud_speak_embed
                else:
                    embeddings += aud_speak_embed
            if self.use_mup:
                embeddings = embeddings * self.input_mult
        # for depth gpt [(s-1)b, c] -> [c, (s-1)b, hd]
        else:
            embeddings = torch.stack(
                [self.audio_embeddings[c](audio_ids[:, c]) for c in range(self.num_channel)]
            )
        return embeddings