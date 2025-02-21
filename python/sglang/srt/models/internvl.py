from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import ImageInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.intern_vit import InternVisionModel
from sglang.srt.models.internlm2 import InternLM2ForCausalLM
from sglang.srt.models.llama import LlamaForCausalLM


class InternVLChatModel(nn.Module):

    def _init_mlp1(self, config: PretrainedConfig) -> nn.Sequential:
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        return nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / config.downsample_ratio) ** 2),
            nn.Linear(
                vit_hidden_size * int(1 / config.downsample_ratio) ** 2, llm_hidden_size
            ),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.vision_tower = InternVisionModel(config.vision_config)

        self.multi_modal_projector = self._init_mlp1(config)
        if config.llm_config.architectures[0] == "LlamaForCausalLM":
            self.language_model = LlamaForCausalLM(config.llm_config)
        elif config.llm_config.architectures[0] == "InternLM2ForCausalLM":
            self.language_model = InternLM2ForCausalLM(config.llm_config)
        self.IMG_CONTEXT_TOKEN_ID = 64000
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.downsample_ratio = config.downsample_ratio
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )

    def pad_input_ids(self, input_ids: List[int], image_inputs: ImageInputs):
        image_sizes, pad_values = image_inputs.image_sizes, image_inputs.pad_values
        offset_list = []
        for idx, image_s in enumerate(image_sizes):
            feature_size = self.num_image_token

            # 构建图像token序列
            pad_ids = pad_values * (
                (
                    feature_size * image_inputs.pixel_values[idx].shape[0]
                    + len(pad_values)
                )
                // len(pad_values)
            )

            # 找到<IMG_CONTEXT>标记的位置
            try:
                offset = input_ids.index(self.IMG_CONTEXT_TOKEN_ID)
            except ValueError:
                offset = 0

            # 替换<IMG_CONTEXT>标记为图像token序列
            input_ids = input_ids[:offset] + pad_ids + input_ids[offset + 1 :]

            offset_list.append(offset)
        image_inputs.image_offsets = offset_list
        return input_ids

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.config.ps_version == "v1":
            pass
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vit_embeds = self.vision_tower(pixel_values=pixel_values)
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.multi_modal_projector(vit_embeds)
        return vit_embeds

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        image_inputs = forward_batch.image_inputs

        if forward_batch.forward_mode.is_extend():
            if self.config.llm_config.architectures[0] == "LlamaForCausalLM":
                input_embeds = self.language_model.model.embed_tokens(input_ids)
            else:
                input_embeds = self.language_model.model.tok_embeddings(input_ids)
            extend_start_loc_cpu = forward_batch.extend_start_loc.cpu().numpy()
            prefix_lens_cpu = forward_batch.extend_prefix_lens.cpu().numpy()
            for i, image in enumerate(image_inputs):
                if image is None:
                    continue
                start_idx = extend_start_loc_cpu[i]
                prefix_len = prefix_lens_cpu[i]
                image_offsets = image.image_offsets
                for idx, image_offset in enumerate(image_offsets):
                    if image_offset < prefix_len:
                        continue
                    pixel_values = image.pixel_values[idx]
                    image_features = self.encode_images(pixel_values)
                    num_image_tokens = image_features.shape[0] * image_features.shape[1]
                    left_idx = start_idx + (image_offset - prefix_len)
                    right_idx = (
                        start_idx + (image_offset - prefix_len) + num_image_tokens
                    )
                    _, C = input_embeds.shape
                    input_embeds[left_idx:right_idx] = image_features[
                        :num_image_tokens
                    ].reshape(-1, C)

            return self.language_model(
                input_ids, positions, forward_batch, input_embeds=input_embeds
            )
        elif forward_batch.forward_mode.is_decode():
            return self.language_model(input_ids, positions, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):

        # load mm_projector
        projector_weights = {
            "mlp1": "multi_modal_projector",
        }
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "mlp1" in name:
                for weight_name, param_name in projector_weights.items():
                    if weight_name in name:
                        name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            elif "vision_model" in name:
                name = name.replace("vision_model.", "")
                self.vision_tower.load_weights([(name, loaded_weight)])
            else:
                name = name.replace("language_model.", "")
                self.language_model.load_weights([(name, loaded_weight)])

    @property
    def num_patches_per_side(self):
        return self.image_size // self.patch_size


EntryClass = [InternVLChatModel]
