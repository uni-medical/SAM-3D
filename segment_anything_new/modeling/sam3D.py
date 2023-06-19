# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder3D import ImageEncoderViT3D
from .mask_decoder3D import MaskDecoder3D
from .prompt_encoder3D import PromptEncoder3D


class Sam3D(nn.Module):
    mask_threshold: float = 0.0
    # image_format: str = "RGB"
    image_format: str = "L"

    def __init__(
        self,
        image_encoder: ImageEncoderViT3D,
        prompt_encoder: PromptEncoder3D,
        mask_decoder: MaskDecoder3D,
        pixel_mean: List[float] = [123.675],  # [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395],  # [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-3:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1], : input_size[2]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        d, h, w = x.shape[-3:]
        padd = self.image_encoder.img_size - d
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh, 0, padd))
        return x


if __name__ == "__main__":
    # iev3d = ImageEncoderViT3D(out_chans=768)
    # md3d = MaskDecoder3D(transformer_dim=768)
    # pe3d = PromptEncoder3D(768, (16,16,16), (256,256,256), 1)
    # model = Sam3D(image_encoder=iev3d, mask_decoder=md3d, prompt_encoder=pe3d)
    # model.eval()
    from functools import partial
    encoder_embed_dim=768
    encoder_depth=12
    encoder_num_heads=12
    encoder_global_attn_indexes=[2, 5, 8, 11]
    prompt_embed_dim = 384
    image_size = 128
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam3D(
        image_encoder=ImageEncoderViT3D(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder3D(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder3D(
            num_multimask_outputs=3,
            # transformer=TwoWayTransformer3D(
            #     depth=2,
            #     embedding_dim=prompt_embed_dim,
            #     mlp_dim=2048,
            #     num_heads=8,
            # ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675],#, 116.28, 103.53],
        pixel_std=[58.395],#, 57.12, 57.375],
    )


    # print(sam)
    x = torch.randn(1, 1, 128, 128, 128)
    y = torch.randn(1, 1, 128, 128, 128)
    low_res_masks = torch.randn(1, 1, 32, 32, 32)
    image_embedding = sam.image_encoder(x)
    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=None,
        boxes=None,
        masks=low_res_masks,
    )
    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=image_embedding, # (B, 256, 64, 64)
        image_pe=sam.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False,
    )
    print(low_res_masks.shape)  # (1, 1, 32, 32, 32)


    # print(sam([{"image":x, "label":y}], False).shape)





# Sam3D(
#   (image_encoder): ImageEncoderViT3D(
#     (patch_embed): PatchEmbed3D(
#       (proj): Conv3d(1, 768, kernel_size=(16, 16, 16), stride=(16, 16, 16))
#     )
#     (blocks): ModuleList(
#       (0-11): 12 x Block3D(
#         (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
#         (attn): Attention(
#           (qkv): Linear(in_features=768, out_features=2304, bias=True)
#           (proj): Linear(in_features=768, out_features=768, bias=True)
#         )
#         (norm2): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
#         (mlp): MLPBlock(
#           (lin1): Linear(in_features=768, out_features=3072, bias=True)
#           (lin2): Linear(in_features=3072, out_features=768, bias=True)
#           (act): GELU(approximate='none')
#         )
#       )
#     )
#     (neck): Sequential(
#       (0): Conv3d(768, 384, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
#       (1): LayerNorm3d()
#       (2): Conv3d(384, 384, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
#       (3): LayerNorm3d()
#     )
#   )
#   (prompt_encoder): PromptEncoder3D(
#     (pe_layer): PositionEmbeddingRandom3D()
#     (point_embeddings): ModuleList(
#       (0-1): 2 x Embedding(1, 384)
#     )
#     (not_a_point_embed): Embedding(1, 384)
#     (mask_downscaling): Sequential(
#       (0): Conv3d(1, 4, kernel_size=(2, 2, 2), stride=(2, 2, 2))
#       (1): LayerNorm3d()
#       (2): GELU(approximate='none')
#       (3): Conv3d(4, 16, kernel_size=(2, 2, 2), stride=(2, 2, 2))
#       (4): LayerNorm3d()
#       (5): GELU(approximate='none')
#       (6): Conv3d(16, 384, kernel_size=(1, 1, 1), stride=(1, 1, 1))
#     )
#     (no_mask_embed): Embedding(1, 384)
#   )
#   (mask_decoder): MaskDecoder3D(
#     (transformer): TwoWayTransformer3D(
#       (layers): ModuleList(
#         (0-1): 2 x TwoWayAttentionBlock3D(
#           (self_attn): Attention(
#             (q_proj): Linear(in_features=384, out_features=384, bias=True)
#             (k_proj): Linear(in_features=384, out_features=384, bias=True)
#             (v_proj): Linear(in_features=384, out_features=384, bias=True)
#             (out_proj): Linear(in_features=384, out_features=384, bias=True)
#           )
#           (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
#           (cross_attn_token_to_image): Attention(
#             (q_proj): Linear(in_features=384, out_features=192, bias=True)
#             (k_proj): Linear(in_features=384, out_features=192, bias=True)
#             (v_proj): Linear(in_features=384, out_features=192, bias=True)
#             (out_proj): Linear(in_features=192, out_features=384, bias=True)
#           )
#           (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
#           (mlp): MLPBlock3D(
#             (lin1): Linear(in_features=384, out_features=2048, bias=True)
#             (lin2): Linear(in_features=2048, out_features=384, bias=True)
#             (act): ReLU()
#           )
#           (norm3): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
#           (norm4): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
#           (cross_attn_image_to_token): Attention(
#             (q_proj): Linear(in_features=384, out_features=192, bias=True)
#             (k_proj): Linear(in_features=384, out_features=192, bias=True)
#             (v_proj): Linear(in_features=384, out_features=192, bias=True)
#             (out_proj): Linear(in_features=192, out_features=384, bias=True)
#           )
#         )
#       )
#       (final_attn_token_to_image): Attention(
#         (q_proj): Linear(in_features=384, out_features=192, bias=True)
#         (k_proj): Linear(in_features=384, out_features=192, bias=True)
#         (v_proj): Linear(in_features=384, out_features=192, bias=True)
#         (out_proj): Linear(in_features=192, out_features=384, bias=True)
#       )
#       (norm_final_attn): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
#     )
#     (iou_token): Embedding(1, 384)
#     (mask_tokens): Embedding(4, 384)
#     (output_upscaling): Sequential(
#       (0): ConvTranspose3d(384, 96, kernel_size=(2, 2, 2), stride=(2, 2, 2))
#       (1): LayerNorm3d()
#       (2): GELU(approximate='none')
#       (3): ConvTranspose3d(96, 48, kernel_size=(2, 2, 2), stride=(2, 2, 2))
#       (4): GELU(approximate='none')
#     )
#     (output_hypernetworks_mlps): ModuleList(
#       (0-3): 4 x MLP(
#         (layers): ModuleList(
#           (0-1): 2 x Linear(in_features=384, out_features=384, bias=True)
#           (2): Linear(in_features=384, out_features=48, bias=True)
#         )
#       )
#     )
#     (iou_prediction_head): MLP(
#       (layers): ModuleList(
#         (0): Linear(in_features=384, out_features=256, bias=True)
#         (1): Linear(in_features=256, out_features=256, bias=True)
#         (2): Linear(in_features=256, out_features=4, bias=True)
#       )
#     )
#   )
# )






















    # print(model)

# Sam3D(
#   (image_encoder): ImageEncoderViT3D(
#     (patch_embed): PatchEmbed3D(
#       (proj): Conv3d(1, 768, kernel_size=(16, 16, 16), stride=(16, 16, 16))
#     )
#     (blocks): ModuleList(
#       (0-11): 12 x Block3D(
#         (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#         (attn): Attention(
#           (qkv): Linear(in_features=768, out_features=2304, bias=True)
#           (proj): Linear(in_features=768, out_features=768, bias=True)
#         )
#         (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#         (mlp): MLPBlock(
#           (lin1): Linear(in_features=768, out_features=3072, bias=True)
#           (lin2): Linear(in_features=3072, out_features=768, bias=True)
#           (act): GELU(approximate='none')
#         )
#       )
#     )
#     (neck): Sequential(
#       (0): Conv3d(768, 768, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
#       (1): LayerNorm3d()
#       (2): Conv3d(768, 768, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
#       (3): LayerNorm3d()
#     )
#   )
#   (prompt_encoder): PromptEncoder3D(
#     (pe_layer): PositionEmbeddingRandom3D()
#     (point_embeddings): ModuleList(
#       (0-1): 2 x Embedding(1, 768)
#     )
#     (not_a_point_embed): Embedding(1, 768)
#     (mask_downscaling): Sequential(
#       (0): Conv3d(1, 0, kernel_size=(2, 2, 2), stride=(2, 2, 2))
#       (1): LayerNorm3d()
#       (2): GELU(approximate='none')
#       (3): Conv3d(0, 1, kernel_size=(2, 2, 2), stride=(2, 2, 2))
#       (4): LayerNorm3d()
#       (5): GELU(approximate='none')
#       (6): Conv3d(1, 768, kernel_size=(1, 1, 1), stride=(1, 1, 1))
#     )
#     (no_mask_embed): Embedding(1, 768)
#   )
#   (mask_decoder): MaskDecoder3D(
#     (transformer): TwoWayTransformer3D(
#       (layers): ModuleList(
#         (0-1): 2 x TwoWayAttentionBlock3D(
#           (self_attn): Attention(
#             (q_proj): Linear(in_features=768, out_features=768, bias=True)
#             (k_proj): Linear(in_features=768, out_features=768, bias=True)
#             (v_proj): Linear(in_features=768, out_features=768, bias=True)
#             (out_proj): Linear(in_features=768, out_features=768, bias=True)
#           )
#           (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#           (cross_attn_token_to_image): Attention(
#             (q_proj): Linear(in_features=768, out_features=384, bias=True)
#             (k_proj): Linear(in_features=768, out_features=384, bias=True)
#             (v_proj): Linear(in_features=768, out_features=384, bias=True)
#             (out_proj): Linear(in_features=384, out_features=768, bias=True)
#           )
#           (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#           (mlp): MLPBlock3D(
#             (lin1): Linear(in_features=768, out_features=2048, bias=True)
#             (lin2): Linear(in_features=2048, out_features=768, bias=True)
#             (act): ReLU()
#           )
#           (norm3): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#           (norm4): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#           (cross_attn_image_to_token): Attention(
#             (q_proj): Linear(in_features=768, out_features=384, bias=True)
#             (k_proj): Linear(in_features=768, out_features=384, bias=True)
#             (v_proj): Linear(in_features=768, out_features=384, bias=True)
#             (out_proj): Linear(in_features=384, out_features=768, bias=True)
#           )
#         )
#       )
#       (final_attn_token_to_image): Attention(
#         (q_proj): Linear(in_features=768, out_features=384, bias=True)
#         (k_proj): Linear(in_features=768, out_features=384, bias=True)
#         (v_proj): Linear(in_features=768, out_features=384, bias=True)
#         (out_proj): Linear(in_features=384, out_features=768, bias=True)
#       )
#       (norm_final_attn): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#     )
#     (iou_token): Embedding(1, 768)
#     (mask_tokens): Embedding(4, 768)
#     (output_upscaling): Sequential(
#       (0): ConvTranspose3d(768, 192, kernel_size=(2, 2, 2), stride=(2, 2, 2))
#       (1): LayerNorm3d()
#       (2): GELU(approximate='none')
#       (3): ConvTranspose3d(192, 96, kernel_size=(2, 2, 2), stride=(2, 2, 2))
#       (4): GELU(approximate='none')
#     )
#     (output_hypernetworks_mlps): ModuleList(
#       (0-3): 4 x MLP(
#         (layers): ModuleList(
#           (0-1): 2 x Linear(in_features=768, out_features=768, bias=True)
#           (2): Linear(in_features=768, out_features=96, bias=True)
#         )
#       )
#     )
#     (iou_prediction_head): MLP(
#       (layers): ModuleList(
#         (0): Linear(in_features=768, out_features=256, bias=True)
#         (1): Linear(in_features=256, out_features=256, bias=True)
#         (2): Linear(in_features=256, out_features=4, bias=True)
#       )
#     )
#   )
# )