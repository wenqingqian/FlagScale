# Copyright (c) 2025, BAAI. All rights reserved.

"""Qwen3.5 colocated MIMO model wrapper."""

from typing import Any, Dict, List

import torch
import torch.distributed as dist

from megatron.core import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from flagscale.models.mimo import switch_parallel_state
from flagscale.models.mimo.mimo_bridge import (
    get_source_vision_rank,
    broadcast_to_language_tp,
    reduce_visual_grad_from_language_tp,
)
from flagscale.models.mimo.mimo_scheduler import (
    MIMOMicrobatchScheduler,
    compute_microbatch_token_counts,
    split_visual_embeds,
    concatenate_visual_grads,
)
from flagscale.models.megatron.qwen35.language_model import Qwen35LanguageModule
from flagscale.models.megatron.qwen35.rope import get_rope_index
from flagscale.models.megatron.qwen35.transformer_config import Qwen35TransformerConfig
from flagscale.models.megatron.qwen3_vl.vision_model import Qwen3VisionModel


class Qwen35MIMOModel(MegatronModule):
    """Qwen3.5 MIMO model with colocated vision and language modules.

    The vision encoder + projection and the language model are constructed as
    separate submodules, each running under its own ``parallel_state`` context.
    """

    def __init__(
        self,
        language_transformer_config: Qwen35TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        vision_transformer_config: TransformerConfig,
        vision_transformer_layer_spec: ModuleSpec,
        vision_projection_config: TransformerConfig,
        vision_projection_layer_spec: ModuleSpec,
        pg_collections: Dict[str, object],
        vision_parallelism,
        language_parallelism,
        vision_projection_type: str = "mlp",
        parallel_output: bool = True,
        language_position_embedding_type: str = "mrope",
        language_rotary_percent: float = 0.25,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        language_rotary_base: int = 10000000,
        fp16_lm_cross_entropy: bool = False,
        language_share_embeddings_and_output_weights: bool = False,
        mtp_block_spec=None,
        vit_batch_factor: int = 1,
        use_fp32_grad_cache: bool = False,
    ) -> None:
        super().__init__(config=language_transformer_config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.pg_collections = pg_collections
        self.vision_pg = pg_collections["vision"]
        self.language_pg = pg_collections["language"]

        self.vision_model = None
        self.language_model = None

        self.vit_batch_factor = max(1, vit_batch_factor)
        self.use_scheduler = self.vit_batch_factor > 1
        self.scheduler = MIMOMicrobatchScheduler(
            vit_batch_factor=self.vit_batch_factor,
            vision_forward_fn=self._vision_forward_fn,
            vision_backward_fn=self._vision_backward_fn,
            use_fp32_grad_cache=use_fp32_grad_cache,
        )

        # Buffers populated during _vision_forward_fn and consumed by _vision_backward_fn.
        self._macro_vision_embeds: torch.Tensor | None = None
        self._macro_deepstack_features: list[torch.Tensor] | None = None

        # Build vision module under the vision parallel context.
        if self.pre_process and self.add_encoder:
            with switch_parallel_state(self.vision_pg):
                self.vision_model = Qwen3VisionModel(
                    vision_transformer_config,
                    vision_transformer_layer_spec,
                    vision_projection_config,
                    vision_projection_layer_spec,
                    projection_type=vision_projection_type,
                    pre_process=True,
                    post_process=True,
                )

        # Build language module under the language parallel context.
        with switch_parallel_state(self.language_pg):
            self.language_model = Qwen35LanguageModule(
                config=language_transformer_config,
                transformer_layer_spec=language_transformer_layer_spec,
                vocab_size=language_vocab_size,
                max_sequence_length=language_max_sequence_length,
                parallel_output=parallel_output,
                position_embedding_type=language_position_embedding_type,
                rotary_percent=language_rotary_percent,
                pre_process=self.pre_process,
                post_process=self.post_process,
                rotary_base=language_rotary_base,
                fp16_lm_cross_entropy=fp16_lm_cross_entropy,
                share_embeddings_and_output_weights=language_share_embeddings_and_output_weights,
                rope_scaling=False,
                mtp_block_spec=mtp_block_spec,
                pg_collection=self.language_pg,
            )

        self.share_embeddings_and_output_weights = (
            self.language_model.share_embeddings_and_output_weights
        )

    def shared_embedding_or_output_weight(self):
        """Surface the language model's word embeddings for gradient all-reduce."""
        if self.add_decoder:
            with switch_parallel_state(self.language_pg):
                return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        with switch_parallel_state(self.language_pg):
            self.language_model.set_input_tensor(input_tensor)

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ):
        """Freeze model modules."""
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_vision_projection and self.vision_model is not None and self.vision_model.projection is not None:
            modules.append(self.vision_model.projection)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def next_microbatch(self, data_iterator, get_batch_fn):
        """Return the next LLM microbatch and its vision output from the scheduler.

        Assembles a new ViT macro batch when the current one is exhausted.  All
        scheduler orchestration lives behind this method so that
        ``forward_step`` never touches the scheduler directly.  The batch dict
        must be returned to the caller (Megatron builds the model inputs and
        the loss closure from it), which is why the advance cannot happen
        inside ``forward``.
        """
        assert self.use_scheduler, "next_microbatch is only valid when the scheduler is on"
        if self.scheduler.need_new_macro_batch():
            self.scheduler.prepare_macro_batch(get_batch_fn, data_iterator, self)
        _, batch, vision_output = self.scheduler.advance()
        return batch, vision_output

    # ------------------------------------------------------------------
    # Scheduler callbacks: run ViT on a macro batch and back-propagate later.
    # ------------------------------------------------------------------
    def _vision_forward_fn(self, batches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run one ViT forward over ``vit_batch_factor`` microbatches.

        ``batches`` is a list of dictionaries returned by the training loop's
        ``get_batch``.  We concatenate visual inputs, run the vision model under
        the vision parallel context, reshard outputs to the language TP layout,
        split them back into per-microbatch chunks, and broadcast each chunk
        inside the target language TP group.
        """
        if not self.pre_process or not self.add_encoder or self.vision_model is None:
            return [{"vision_embeds": None, "deepstack_features": None} for _ in batches]

        vision_tp_size = dist.get_world_size(self.vision_pg.tp)
        assert vision_tp_size == 1, (
            f"MIMO vision backward currently supports vision TP=1 only; got {vision_tp_size}."
        )

        # ``get_batch`` returns separate image/video tensors; concatenate them
        # along the sample dimension to match the ViT input layout.
        def _cat_imgs_videos(b):
            imgs = b.get("imgs")
            videos = b.get("videos")
            if imgs is None:
                return videos
            if videos is None:
                return imgs
            return torch.cat([imgs, videos], dim=0)

        def _cat_grids(b):
            image_grids = b.get("image_thw_grids")
            video_grids = b.get("video_thw_grids")
            if image_grids is None:
                return video_grids
            if video_grids is None:
                return image_grids
            return torch.cat([image_grids, video_grids], dim=0)

        per_batch_imgs_videos = [_cat_imgs_videos(b) for b in batches]
        per_batch_grids = [_cat_grids(b) for b in batches]

        valid_imgs_videos = [t for t in per_batch_imgs_videos if t is not None]
        valid_grids = [t for t in per_batch_grids if t is not None]

        if not valid_grids:
            return [{"vision_embeds": None, "deepstack_features": None} for _ in batches]

        vision_data = torch.cat(valid_imgs_videos, dim=0)
        vision_grid_thw = torch.cat(valid_grids, dim=0)

        with switch_parallel_state(self.vision_pg):
            if vision_grid_thw.shape[0] == 0:
                macro_embeds = None
                macro_deepstack = None
            else:
                macro_embeds, macro_deepstack = self.vision_model(
                    vision_data=vision_data,
                    grid_thw=vision_grid_thw,
                )

        # Vision outputs are full [T, H] on every ViT TP rank; keep them full
        # and only broadcast inside the target language TP group.
        self._macro_vision_embeds = macro_embeds
        self._macro_deepstack_features = macro_deepstack

        # Account for Qwen3-VL spatial merge when splitting the macro output.
        spatial_merge_unit = getattr(self.vision_model, "spatial_merge_unit", 1)
        token_counts = compute_microbatch_token_counts(
            per_batch_grids, merge_unit=spatial_merge_unit
        )
        split_outputs = split_visual_embeds(macro_embeds, macro_deepstack, token_counts, dim=0)

        # Broadcast each microbatch's visual embeds inside its language TP group.
        # Source rank cycles through the ViT ranks that overlap this language DP replica.
        for forward_idx, output in enumerate(split_outputs):
            src_rank = get_source_vision_rank(self.language_pg, forward_idx, self.vit_batch_factor)
            if output["vision_embeds"] is not None:
                output["vision_embeds"] = broadcast_to_language_tp(
                    output["vision_embeds"], self.language_pg, src_rank
                )
            if output["deepstack_features"] is not None:
                output["deepstack_features"] = [
                    broadcast_to_language_tp(f, self.language_pg, src_rank)
                    for f in output["deepstack_features"]
                ]

        return split_outputs

    def _vision_backward_fn(self, gradients: List[Dict[str, Any]]) -> None:
        """Run ViT backward after all microbatch gradients are collected.

        Gradients live on detached visual tensors, so we manually concatenate
        them and invoke ``.backward()`` on the original ViT outputs.  Each
        microbatch gradient is first reduced to its source rank and then
        gathered so every rank holds the full gradient for its own ViT shard.
        """
        if not self.pre_process or not self.add_encoder or self.vision_model is None:
            return

        macro_embeds = self._macro_vision_embeds
        macro_deepstack = self._macro_deepstack_features
        if macro_embeds is None:
            return

        num_chunks = len(gradients)
        source_ranks = [
            get_source_vision_rank(self.language_pg, i, self.vit_batch_factor)
            for i in range(num_chunks)
        ]

        # The current gradient assembly gathers the full macro gradient on every
        # rank.  This is correct only when the vision module is not tensor-parallel
        # (TP=1); with vision TP > 1 each rank only needs its own gradient shard.
        assert dist.get_world_size(self.vision_pg.tp) == 1, (
            "vision gradient assembly assumes vision TP=1; TP > 1 requires "
            "shard-aware reduce/gather instead of full gradient broadcast"
        )

        def _assemble_full_grad(key: str) -> torch.Tensor:
            # Microbatches without visual data contribute ``None`` gradient dicts.
            # Skip them; their token count is zero so the concatenated gradient
            # still matches the original macro output.
            items = [
                (g[key], src)
                for g, src in zip(gradients, source_ranks)
                if g is not None and key in g
            ]
            if not items:
                return None
            chunks, ranks = zip(*items)
            reduced = [
                reduce_visual_grad_from_language_tp(chunk, self.language_pg, src)
                for chunk, src in zip(chunks, ranks)
            ]
            gathered = [
                broadcast_to_language_tp(chunk, self.language_pg, src)
                for chunk, src in zip(reduced, ranks)
            ]
            return concatenate_visual_grads(
                [{key: g} for g in gathered], key=key, dim=0
            )

        embed_grad = _assemble_full_grad("vision_embeds")
        if embed_grad is None:
            return

        # Cast gradients back to the ViT output dtype when fp32 cache was used.
        target_dtype = macro_embeds.dtype
        def _to_vit_dtype(g):
            return g.to(target_dtype) if g is not None else None

        grads = [_to_vit_dtype(embed_grad)]
        if macro_deepstack is not None:
            for i in range(len(macro_deepstack)):
                grads.append(_to_vit_dtype(_assemble_full_grad(f"deepstack_{i}")))

        targets = [macro_embeds] + (macro_deepstack or [])
        with switch_parallel_state(self.vision_pg):
            torch.autograd.backward(targets, grads)

        self._macro_vision_embeds = None
        self._macro_deepstack_features = None

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        vision_data: torch.Tensor = None,
        vision_grid_thw: torch.Tensor = None,
        video_start_index: int = -1,
        image_input_mask: torch.Tensor = None,
        video_input_mask: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        loss_mask: torch.Tensor | None = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        vision_output: dict = None,
    ) -> torch.Tensor:
        """Forward function of Qwen3.5 MIMO model."""
        use_inference_kv_cache = (
            inference_params is not None
            and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        if use_inference_kv_cache:
            raise NotImplementedError()

        if self.pre_process and self.add_encoder:
            if vision_output is not None:
                # Scheduler path: caller already advanced the scheduler.
                vision_embeds = vision_output["vision_embeds"]
                deepstack_feature_lists = vision_output["deepstack_features"]
            elif self.use_scheduler:
                # Model-internal advance: a caller prepared the macro batch but
                # left advancing to the model (not used by forward_step, which
                # needs the batch up front and therefore advances externally).
                _, _, vision_output = self.scheduler.advance()
                vision_embeds = vision_output["vision_embeds"]
                deepstack_feature_lists = vision_output["deepstack_features"]
            else:
                # Direct path (vit_batch_factor == 1, e.g. vision_micro_batch_size
                # == language micro batch or vision TP == language TP): run ViT
                # for this single microbatch.  Not used when the scheduler is on.
                with switch_parallel_state(self.vision_pg):
                    vision_embeds = None
                    deepstack_feature_lists = None
                    if vision_grid_thw is not None and vision_grid_thw.shape[0] > 0:
                        vision_embeds, deepstack_feature_lists = self.vision_model(
                            vision_data=vision_data,
                            grid_thw=vision_grid_thw,
                        )
                    else:
                        vision_embeds = None
                        deepstack_feature_lists = None

                # Vision outputs are full [T, H] tensors on every ViT TP rank.
                # Broadcast them inside the language TP group so every language
                # rank sees the same visual embeddings, matching the baseline
                # behaviour where the embedding layer consumes full tensors.
                src_rank = get_source_vision_rank(
                    self.language_pg, 0, self.vit_batch_factor
                )
                if vision_embeds is not None:
                    vision_embeds = broadcast_to_language_tp(
                        vision_embeds, self.language_pg, src_rank
                    )
                    if deepstack_feature_lists is not None:
                        deepstack_feature_lists = [
                            broadcast_to_language_tp(f, self.language_pg, src_rank)
                            for f in deepstack_feature_lists
                        ]

            # Detach visual tensors from the ViT graph and capture gradients for
            # delayed ViT backward when the scheduler is active.
            if self.use_scheduler:
                vision_embeds = self.scheduler.register_visual_grad_hook(
                    vision_embeds, "vision_embeds"
                )
                if deepstack_feature_lists is not None:
                    deepstack_feature_lists = [
                        self.scheduler.register_visual_grad_hook(f, f"deepstack_{i}")
                        for i, f in enumerate(deepstack_feature_lists)
                    ]

            if inference_params is not None:
                raise NotImplementedError()

            with switch_parallel_state(self.language_pg):
                if use_inference_kv_cache:
                    language_embeddings = self.language_model.embedding(
                        input_ids=input_ids,
                        position_ids=None,
                    )
                    combined_embeddings = language_embeddings
                elif vision_embeds is not None:
                    if image_input_mask is not None:
                        image_input_mask = image_input_mask.T
                    if video_input_mask is not None:
                        video_input_mask = video_input_mask.T

                    if video_start_index == 0:
                        image_embeds = None
                        video_embeds = vision_embeds
                        visual_pos_masks = video_input_mask
                    elif video_start_index == vision_embeds.shape[0]:
                        image_embeds = vision_embeds
                        video_embeds = None
                        visual_pos_masks = image_input_mask
                    elif 0 < video_start_index < vision_embeds.shape[0]:
                        image_embeds = vision_embeds[:video_start_index]
                        video_embeds = vision_embeds[video_start_index:]
                        visual_pos_masks = torch.logical_or(image_input_mask, video_input_mask)
                    else:
                        raise ValueError(
                            f"Expect video token start index in range [0, {vision_embeds.shape[0]}], "
                            f"but got {video_start_index}"
                        )

                    combined_embeddings = self.language_model.embedding(
                        input_ids=input_ids,
                        position_ids=None,
                        image_input_mask=image_input_mask,
                        video_input_mask=video_input_mask,
                        image_embeds=image_embeds,
                        video_embeds=video_embeds,
                    )
                else:
                    combined_embeddings = self.language_model.embedding(
                        input_ids=input_ids,
                        position_ids=None,
                    )
                    visual_pos_masks = None
                    deepstack_feature_lists = None
        else:
            combined_embeddings = None
            visual_pos_masks = None
            deepstack_feature_lists = None

        with switch_parallel_state(self.language_pg):
            output = self.language_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=combined_embeddings,
                labels=labels,
                loss_mask=loss_mask,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_feature_lists,
                **(extra_block_kwargs or {}),
            )
        return output

    def get_rope_index(
        self,
        input_ids: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mRoPE position indices for Qwen3.5."""
        return get_rope_index(
            spatial_merge_size=self.config.spatial_merge_size,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
