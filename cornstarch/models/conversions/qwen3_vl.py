from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    BaseModelOutputWithDeepstackFeatures,
    Qwen3VLVisionModel,
)

from cornstarch.models.forward_specs import (
    LayerContext,
    TransformerForwardSpec,
    _filtered_layer_kwargs,
)
from cornstarch.models.vision_encoder import CornstarchVisionEncoder


@dataclass(frozen=True)
class Qwen3VLVisionForwardSpec(TransformerForwardSpec):
    """Native forward spec for Qwen3-VL vision encoders."""

    num_grid_per_side: int
    spatial_merge_size: int
    deepstack_visual_indexes: tuple[int, ...]

    def embed_inputs(self, model: nn.Module, **kwargs: Any) -> torch.Tensor:
        hidden_states = model.pre_encoder["patch_embed"](kwargs["hidden_states"])
        pos_embeds = self._fast_pos_embed_interpolate(model, kwargs["grid_thw"])
        return hidden_states + pos_embeds

    def prepare_layer_context(
        self, model: nn.Module, hidden_states: torch.Tensor, **kwargs: Any
    ) -> LayerContext:
        grid_thw = kwargs["grid_thw"]
        rotary_pos_emb = self._rot_pos_emb(model, grid_thw)
        seq_len, _ = hidden_states.size()
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (
            emb.float().cos().to(dtype=hidden_states.dtype),
            emb.float().sin().to(dtype=hidden_states.dtype),
        )
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        return {
            "cu_seqlens": F.pad(cu_seqlens, (1, 0), value=0),
            "deepstack_features": [],
            "position_embeddings": position_embeddings,
        }

    def get_layer_kwargs(
        self, model: nn.Module, layer_idx: int, context: LayerContext, **kwargs: Any
    ) -> dict[str, Any]:
        return {
            "cu_seqlens": context["cu_seqlens"],
            "position_embeddings": context["position_embeddings"],
            **_filtered_layer_kwargs(kwargs),
        }

    def process_layer_output(
        self,
        model: nn.Module,
        layer_idx: int,
        layer_output: Any,
        context: LayerContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = super().process_layer_output(
            model, layer_idx, layer_output, context, **kwargs
        )
        if layer_idx in self.deepstack_visual_indexes:
            merger_index = self.deepstack_visual_indexes.index(layer_idx)
            deepstack_feature = model.post_encoder["deepstack_merger_list"][merger_index](
                hidden_states
            )
            context["deepstack_features"].append(deepstack_feature)
        return hidden_states

    def build_output(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> BaseModelOutputWithDeepstackFeatures:
        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=model.post_encoder["merger"](hidden_states),
            deepstack_features=context["deepstack_features"],
        )

    def _fast_pos_embed_interpolate(
        self, model: nn.Module, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        grid_thw_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]
        device = model.pre_encoder["pos_embed"].weight.device
        idx_list: list[list[int]] = [[] for _ in range(4)]
        weight_list: list[list[float]] = [[] for _ in range(4)]

        for _, height, width in grid_thw_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, height)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, width)
            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor
            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for index in range(4):
                idx_list[index].extend(indices[index].tolist())
                weight_list[index].extend(weights[index].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(
            weight_list,
            dtype=model.pre_encoder["pos_embed"].weight.dtype,
            device=device,
        )
        pos_embeds = model.pre_encoder["pos_embed"](idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        for pos_embed, frames, height, width in zip(
            patch_pos_embeds, grid_ts, grid_hs, grid_ws, strict=True
        ):
            pos_embed = pos_embed.repeat(frames, 1)
            pos_embed = (
                pos_embed.view(
                    frames,
                    height // self.spatial_merge_size,
                    self.spatial_merge_size,
                    width // self.spatial_merge_size,
                    self.spatial_merge_size,
                    -1,
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute)

    def _rot_pos_emb(self, model: nn.Module, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_thw_list = grid_thw.tolist()
        max_hw = max(max(height, width) for _, height, width in grid_thw_list)
        freq_table = model.pre_encoder["rotary_pos_emb"](max_hw)
        device = freq_table.device
        total_tokens = sum(frames * height * width for frames, height, width in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw_list:
            merged_h = height // self.spatial_merge_size
            merged_w = width // self.spatial_merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(self.spatial_merge_size, device=device)
            intra_col = torch.arange(self.spatial_merge_size, device=device)
            row_idx = block_rows[:, None, None, None] * self.spatial_merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * self.spatial_merge_size + intra_col[None, None, None, :]
            row_idx = row_idx.expand(
                merged_h, merged_w, self.spatial_merge_size, self.spatial_merge_size
            ).reshape(-1)
            col_idx = col_idx.expand(
                merged_h, merged_w, self.spatial_merge_size, self.spatial_merge_size
            ).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        return freq_table[pos_ids].flatten(1)


def convert_qwen3_vl_vision_config(
    config: Qwen3VLVisionConfig,
    attn_implementation: str | None = None,
    layer_offload_config=None,
    layer_compile_config=None,
) -> CornstarchVisionEncoder:
    """Convert a Qwen3-VL vision config into a Cornstarch vision encoder."""
    with torch.device("meta"):
        hf_model = Qwen3VLVisionModel(copy.deepcopy(config))
    return CornstarchVisionEncoder(
        config,
        pre_encoder={
            "patch_embed": hf_model.patch_embed,
            "pos_embed": hf_model.pos_embed,
            "rotary_pos_emb": hf_model.rotary_pos_emb,
        },
        encoder_layers=hf_model.blocks,
        post_encoder={
            "merger": hf_model.merger,
            "deepstack_merger_list": hf_model.deepstack_merger_list,
        },
        hf_to_cornstarch_prefixes=(
            ("patch_embed.", "pre_encoder.patch_embed."),
            ("pos_embed.", "pre_encoder.pos_embed."),
            ("rotary_pos_emb.", "pre_encoder.rotary_pos_emb."),
            ("blocks.", "encoder_layers."),
            ("merger.", "post_encoder.merger."),
            ("deepstack_merger_list.", "post_encoder.deepstack_merger_list."),
        ),
        hf_model_factory=Qwen3VLVisionModel,
        forward_spec=Qwen3VLVisionForwardSpec(
            num_grid_per_side=hf_model.num_grid_per_side,
            spatial_merge_size=hf_model.spatial_merge_size,
            deepstack_visual_indexes=tuple(hf_model.deepstack_visual_indexes),
        ),
        attn_implementation=attn_implementation,
        layer_offload_config=layer_offload_config,
        layer_compile_config=layer_compile_config,
    )
