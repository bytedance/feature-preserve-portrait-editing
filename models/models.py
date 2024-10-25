# Copyright (c) 2023 The HuggingFace Team (diffusers)
# Copyright (c) 2023 Johanna Karras (DreamPose)
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# This file has been modified by Bytedance Ltd. and/or its affiliates on October 24, 2024.

# Original file (diffusers) was released under Apache License 2.0, with the full license text
# available at https://github.com/huggingface/diffusers/blob/main/LICENSE.

# Original file (DreamPose) was released under MIT License, with the full license text
# available at https://github.com/johannakarras/DreamPose/blob/main/LICENSE.

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import einsum
import torch.utils.checkpoint
from einops import rearrange

import math

from diffusers import AutoencoderKL
from diffusers.models import UNet2DConditionModel

def get_unet(pretrained_model_name_or_path, revision, additional_channel, resolution=256, n_poses=5,random_init=False):
    # Load pretrained UNet layers

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
    )

    # Modify input layer to have 1 additional input channels (pose)
    weights = unet.conv_in.weight.clone()
    unet.conv_in = nn.Conv2d(4 + additional_channel, weights.shape[0], kernel_size=unet.conv_in.kernel_size, padding=unet.conv_in.padding)
    with torch.no_grad():
        unet.conv_in.weight[:, :4] = weights # original weights
        unet.conv_in.weight[:, 4:] = torch.zeros(unet.conv_in.weight[:, 4:].shape) # new weights initialized to zero

    return unet

'''
    This module takes in CLIP + VAE embeddings and outputs CLIP-compatible embeddings.
'''
class Embedding_Adapter(nn.Module):
    def __init__(self,  output_nc=4, norm_layer=nn.InstanceNorm2d, chkpt=None):
        super(Embedding_Adapter, self).__init__()

        self.save_method_name = "adapter"

        self.pool =  nn.MaxPool2d(2)
        self.vae2clip = nn.Linear(1024, 768)

        self.linear1 = nn.Linear(54, output_nc) # 50 x 54 shape

        # initialize weights
        with torch.no_grad():
            self.linear1.weight = nn.Parameter(torch.eye(output_nc, 54))

        if chkpt is not None:
            pass

    def forward(self, clip, vae):
        
        vae = self.pool(vae) # 1 4 80 64 --> 1 4 40 32
        vae = rearrange(vae, 'b c h w -> b c (h w)') # 1 4 20 16 --> 1 4 1280
     
        vae = self.vae2clip(vae) # 1 4 768

        # Concatenate
        concat = torch.cat((clip, vae), 1)

        # Encode

        # print(concat.shape)

        concat = rearrange(concat, 'b c d -> b d c')
        # print(concat.shape)

        concat = self.linear1(concat)
        # print(concat.shape)

        concat = rearrange(concat, 'b d c -> b c d')

        # print('ddd') 
        return concat
