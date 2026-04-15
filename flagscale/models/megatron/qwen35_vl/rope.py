# Copyright (c) 2025, BAAI. All rights reserved.
#
# Adopted from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Qwen3.5 VL mRoPE implementation.

Key differences from Qwen3-VL:
- mrope_section = [11, 11, 10] (vs [24, 20, 20])
- rotary_percent = 0.25 (partial rotary, head_dim=256, rotary_dim=64)
- rotary_base = 10,000,000 (vs 5,000,000)
"""

from typing import List, Optional

import torch
import torch.nn as nn
from megatron.core.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
    get_pos_emb_on_this_cp_rank,
)
from megatron.core.packed_seq_params import PackedSeqParams
from torch import Tensor


class Qwen35VLLanguageRotaryEmbedding(nn.Module):
    """Multimodal Rotary Embedding for Qwen3.5 VL language model.

    Args:
        kv_channels: Number of key/value channels (head_dim).
        rotary_percent: Percent of rotary dimension to use.
        rotary_interleaved: Not supported for Qwen3.5 VL.
        seq_len_interpolation_factor: RoPE interpolation factor.
        rotary_base: Base for rotary position embeddings.
        cp_group: Context parallel process group.
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float = 0.25,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: Optional[float] = None,
        rotary_base: int = 10000000,
        cp_group: torch.distributed.ProcessGroup = None,
    ) -> None:
        super().__init__()

        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.rotary_interleaved = rotary_interleaved
        assert not self.rotary_interleaved, "Qwen3.5 VL only supports non-interleaved rotary"

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.inv_freq = 1.0 / (
            rotary_base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=torch.cuda.current_device()) / dim)
        )
        self.is_thd_format = False
        self.mrope_section = [11, 11, 10]  # Default for Qwen3.5 VL
        assert cp_group is not None, "cp_group is required"
        self.cp_group = cp_group

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.

        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.

        Args:
            freqs: Shape (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,) - section sizes for [temporal, height, width]

        Returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # Overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(
        self,
        position_ids: torch.Tensor,
        mrope_section: List[int] | None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass of multimodal RoPE embedding.

        Args:
            position_ids: Shape [3, batchsize, seqlens]
            mrope_section: [temporal, height, width] section sizes
            packed_seq_params: Packed sequence params for THD format

        Returns:
            Tensor: Embeddings of shape (seq_length, bs, 1, 2 * dim)
        """
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        seq = position_ids.to(device=self.inv_freq.device, dtype=torch.float32)

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        # shape (3, bs, dim, 1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].expand(3, seq.shape[1], -1, 1)
        # shape (3, bs, 1, seq_length)
        seq_expanded = seq[:, :, None, :].float()
        # shape (3, bs, seq_length, dim)
        freqs = (inv_freq_expanded @ seq_expanded).transpose(2, 3)

        if mrope_section is not None:
            freqs = self.apply_interleaved_mrope(freqs, mrope_section)
        else:
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)

        emb = torch.cat((freqs, freqs), dim=-1)

        # shape (seq_length, bs, 1, 2 * dim)
        emb = emb[..., None, :].transpose(0, 1).contiguous()
        if self.cp_group.size() > 1 and not self.is_thd_format:
            emb = get_pos_emb_on_this_cp_rank(emb, 0, self.cp_group)
        return emb


def get_rope_index(
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mRoPE position indices for Qwen3.5 VL.

    Identical algorithm to Qwen3-VL but uses Qwen3.5 token IDs.
    Video timestamps are split the same way as Qwen3-VL.
    """

    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    if packed_seq_params is not None and attention_mask is None and input_ids is not None:
        cu_seqlens = packed_seq_params.cu_seqlens_q
        if cu_seqlens is not None and cu_seqlens.numel() >= 2:
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            attention_mask = torch.zeros_like(input_ids, dtype=input_ids.dtype)
            max_len = attention_mask.shape[1]
            for i, seq_len in enumerate(seq_lens.tolist()):
                valid = min(int(seq_len), max_len)
                attention_mask[i, :valid] = 1
        else:
            attention_mask = torch.ones_like(input_ids)

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        elif attention_mask.dim() > 2:
            attention_mask = attention_mask.any(dim=-1)
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.squeeze(1)
            attention_mask = attention_mask.to(dtype=total_input_ids.dtype)

        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)

        for i, sample_input_ids in enumerate(total_input_ids):
            sample_input_ids = sample_input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(sample_input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = sample_input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = sample_input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=total_input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            if attention_mask.dim() > 2:
                attention_mask = attention_mask.any(dim=-1)
                if attention_mask.dim() == 3:
                    attention_mask = attention_mask.squeeze(1)
                attention_mask = attention_mask.to(dtype=torch.long)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


def apply_rotary_pos_emb_thd_absolute(
    t: Tensor, cu_seqlens: Tensor, freqs: Tensor, rotary_interleaved: bool = False
) -> Tensor:
    """Apply RoPE for THD (packed sequence) format."""
    return _apply_rotary_pos_emb_bshd(t[:, None], freqs, rotary_interleaved=rotary_interleaved).squeeze(1)


def apply_rotary_pos_emb_absolute(
    t: Tensor,
    freqs: Tensor,
    config,
    cu_seqlens: Optional[Tensor] = None,
):
    """Apply absolute RoPE for mRoPE.

    In Qwen3.5 VL, the shape of freqs is (seq_length, bs, 1, 2*dim)
    instead of (max_seqlen, 1, 1, 2*dim).
    """
    assert not config.apply_rope_fusion
    orig_t_dtype = t.dtype
    if config.apply_rotary_pos_emb_in_fp32:
        t = t.float()

    if cu_seqlens is None:
        result = _apply_rotary_pos_emb_bshd(t, freqs, rotary_interleaved=config.rotary_interleaved)
    else:
        result = apply_rotary_pos_emb_thd_absolute(t, cu_seqlens, freqs, rotary_interleaved=config.rotary_interleaved)

    if config.apply_rotary_pos_emb_in_fp32:
        result = result.to(orig_t_dtype)

    return result
