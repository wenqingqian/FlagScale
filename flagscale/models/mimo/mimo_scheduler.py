# Copyright (c) 2025, BAAI. All rights reserved.

"""Microbatch scheduler for colocated MIMO deployment.

When the vision module has a larger effective batch size than the language
module, one ViT forward can feed multiple LLM forwards.  This scheduler hides
the macro-batch buffering and gradient routing from the outer Megatron loop.

Design notes
------------
* The scheduler is model-agnostic: it stores opaque batch dictionaries and
  relies on the owning model to provide ``vision_forward_fn`` and
  ``vision_backward_fn`` callbacks.
* Split visual tensors are detached before being injected into the language
  model.  This keeps the ViT computation graph alive without using
  ``retain_graph=True`` during LLM backward.
* Gradients accumulated on the detached splits are concatenated and handed
  back to ``vision_backward_fn`` once every microbatch in the macro batch has
  produced a gradient.
"""

from collections.abc import Callable
from typing import Any

import torch


class MIMOMicrobatchScheduler:
    """Manage ViT macro batches and route gradients back to ViT.

    Args:
        vit_batch_factor: Number of LLM microbatches served by one ViT forward.
        vision_forward_fn: Callable accepting a list of microbatch dictionaries
            and returning a list of per-microbatch vision outputs.  Each output
            is an opaque dictionary; the model decides what to put inside.
        vision_backward_fn: Callable accepting a list of per-microbatch
            gradients and running ViT backward.  Invoked automatically once
            every microbatch in the macro batch has produced a gradient.
    """

    def __init__(
        self,
        vit_batch_factor: int,
        vision_forward_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
        vision_backward_fn: Callable[[list[dict[str, Any]]], None],
        use_fp32_grad_cache: bool = False,
    ) -> None:
        if vit_batch_factor < 1:
            raise ValueError(f"vit_batch_factor must be >= 1, got {vit_batch_factor}")
        self.vit_batch_factor = vit_batch_factor
        self.vision_forward_fn = vision_forward_fn
        self.vision_backward_fn = vision_backward_fn
        self.use_fp32_grad_cache = use_fp32_grad_cache

        self._macro_batches: list[dict[str, Any]] = []
        self._vision_outputs: list[dict[str, Any]] = []
        self._gradients: list[dict[str, Any] | None] = []
        self._expected_keys: list[set] = []
        self._next_idx: int = 0

    # ------------------------------------------------------------------
    # Macro-batch lifecycle.
    # ------------------------------------------------------------------
    def need_new_macro_batch(self) -> bool:
        """Return True when a new ViT macro batch must be assembled."""
        return self._next_idx >= len(self._macro_batches)

    def prepare_macro_batch(
        self,
        get_batch_fn: Callable,
        data_iterator: Any,
        model: Any,
    ) -> None:
        """Collect ``vit_batch_factor`` microbatches and run ViT forward.

        The caller (usually ``forward_step``) is responsible for providing a
        ``get_batch_fn`` that yields one LLM microbatch per call.
        """
        assert self.need_new_macro_batch(), "scheduler still has unconsumed microbatches"

        batches: list[dict[str, Any]] = []
        for _ in range(self.vit_batch_factor):
            batches.append(get_batch_fn(data_iterator, model))

        outputs = self.vision_forward_fn(batches)
        if not isinstance(outputs, (list, tuple)) or len(outputs) != len(batches):
            raise RuntimeError(
                f"vision_forward_fn must return one output dict per microbatch, "
                f"got {len(outputs)} outputs for {len(batches)} batches"
            )

        self._macro_batches = batches
        self._vision_outputs = list(outputs)
        self._gradients = [None] * len(batches)
        self._expected_keys = [set() for _ in range(len(batches))]
        self._next_idx = 0

    def advance(self) -> tuple[int, dict[str, Any], dict[str, Any]]:
        """Return the next microbatch index, batch dict, and vision output dict."""
        if self.need_new_macro_batch():
            raise RuntimeError(
                "scheduler has no active macro batch; call prepare_macro_batch first"
            )
        idx = self._next_idx
        assert 0 <= idx < len(self._macro_batches), (
            f"advance index {idx} out of range [0, {len(self._macro_batches)})"
        )
        self._next_idx += 1
        return idx, self._macro_batches[idx], self._vision_outputs[idx]

    # ------------------------------------------------------------------
    # Gradient collection.
    # ------------------------------------------------------------------
    def register_visual_grad_hook(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        """Detach ``tensor`` and register a hook that collects its gradient.

        ``key`` names the gradient slot (canonical keys are ``"main"`` and
        ``"aux_{i}"``, assigned by the model wrapper).  The returned tensor is
        safe to feed into the language model: its gradient will be captured by
        the scheduler instead of propagating back into the ViT graph.
        """
        if tensor is None:
            return None
        idx = self._next_idx - 1
        assert idx >= 0, (
            "register_visual_grad_hook called before scheduler.advance(); "
            "no active microbatch index"
        )
        assert idx < len(self._expected_keys), (
            f"microbatch index {idx} out of range [0, {len(self._expected_keys)})"
        )
        assert key not in self._expected_keys[idx], (
            f"gradient hook for key '{key}' already registered for microbatch {idx}"
        )
        tensor = tensor.detach().requires_grad_(tensor.requires_grad)
        if tensor.requires_grad:
            self._expected_keys[idx].add(key)
            tensor.register_hook(lambda grad: self._collect_grad(grad, idx, key))
        return tensor

    def _collect_grad(self, grad: torch.Tensor, idx: int, key: str) -> None:
        """Store one gradient slice and trigger ViT backward when complete."""
        assert 0 <= idx < len(self._gradients), (
            f"gradient index {idx} out of range [0, {len(self._gradients)})"
        )
        if self._gradients[idx] is None:
            self._gradients[idx] = {}
        # Accumulate gradients in fp32 when requested to reduce bf16 round-off.
        if self.use_fp32_grad_cache and grad is not None:
            grad = grad.float()
        self._gradients[idx][key] = grad

        if self._all_gradients_ready():
            self.vision_backward_fn(self._gradients)
            self._reset_after_backward()

    def _all_gradients_ready(self) -> bool:
        """Check whether every registered key for every consumed microbatch is present.

        ViT backward must not run until the whole macro batch has been forwarded,
        otherwise unconsumed microbatches look "ready" because no keys have been
        registered for them yet.
        """
        if self._next_idx < len(self._macro_batches):
            return False
        for i, expected in enumerate(self._expected_keys):
            if not expected:
                # No tensor was registered for this microbatch; treat as ready.
                continue
            grads = self._gradients[i]
            if grads is None or set(grads.keys()) != expected:
                return False
        return True

    def _reset_after_backward(self) -> None:
        """Clear buffers after ViT backward has run."""
        self._macro_batches = []
        self._vision_outputs = []
        self._gradients = []
        self._expected_keys = []
        self._next_idx = 0
