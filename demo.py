# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import os
import torch
import sys
sys.path.append('../')
import random
import numpy as np
import argparse
from collections import OrderedDict
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel, AutoTokenizer, PretrainedConfig
from diffusers import UniPCMultistepScheduler
from pipelines.pipeline import StableDiffusionImg2ImgPipeline
from models.models import get_unet, Embedding_Adapter

def load_unet(model_dir, resolution=512):
    pretrained_model_path = 'stablediffusionapi/realistic-vision-v13'
    unet = get_unet(pretrained_model_path, "", additional_channel=4, resolution=resolution)
    
    unet_path = os.path.join(model_dir, 'unet.pth')
    unet_state_dict = torch.load(unet_path)
    new_state_dict = OrderedDict()
    for k, v in unet_state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    unet.load_state_dict(new_state_dict)
    return unet.cuda()

def import_model_class(pretrained_model_path):
    config = PretrainedConfig.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    model_class = config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def load_pipeline(args, unet):
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float16)
    clip_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", torch_dtype=torch.float16).to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer", use_fast=False)
    
    text_encoder_cls = import_model_class(args.pretrained_model_path)
    text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.pretrained_model_path, 
        unet=unet, 
        image_encoder=clip_encoder, 
        clip_processor=clip_processor,
        torch_dtype=torch.float16
    ).to('cuda')

    return pipe

def load_adapter(pipe, model_dir):
    adapter_chkpt = os.path.join(model_dir, 'adapter.pth')
    adapter_state_dict = torch.load(adapter_chkpt)
    new_state_dict = OrderedDict()
    for k, v in adapter_state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v

    pipe.adapter = Embedding_Adapter(output_nc=50)
    pipe.adapter.load_state_dict(new_state_dict)
    return pipe.adapter.cuda()

def process_image(args, pipe, seeds):
    image = Image.open(args.image_path).resize((512, 512)).convert('RGB')
    out_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.image_path))[0], args.prompt.replace(' ', '_').replace(',', ''))
    os.makedirs(out_path, exist_ok=True)

    for s1 in args.s1s:
        for s2 in args.s2s:
            for s3 in args.s3s:
                for seed in seeds:
                    latents = torch.randn(1, 4, 64, 64).cuda()
                    generator = torch.Generator('cuda').manual_seed(seed)
                    with torch.autocast("cuda"):
                        with torch.no_grad():
                            image_out = pipe(
                                latents=latents, 
                                prompt=args.prompt, 
                                s1=s1, s2=s2, s3=s3, 
                                num_inference_steps=args.inference_steps, 
                                image=image, 
                                generator=generator, 
                                strength=args.strength, 
                                combine_type='concat'
                            ).images[0]

                    # Concatenate original and generated images
                    image_vis = Image.new('RGB', (1024, 512))
                    image_vis.paste(image, (0, 0))
                    image_vis.paste(image_out, (512, 0))
                    image_vis.save(os.path.join(out_path, f'{seed}_{s1}_{s2}_{s3}.jpg'))

def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion Img2Img Pipeline with Configurable Parameters")
    parser.add_argument('--pretrained_model_path', type=str, default='stablediffusionapi/realistic-vision-v13', help="Path to the pretrained model")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory containing the model checkpoints")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image")
    parser.add_argument('--prompt', type=str, default='a man, cute flower costume', help="Text prompt for generation")
    parser.add_argument('--output_dir', type=str, default='demo_results', help="Directory to save the generated images")
    parser.add_argument('--seeds', type=int, nargs='+', default=[1,2,3,4,5], help="List of seeds for generation")
    parser.add_argument('--s1s', type=int, nargs='+', default=[3], help="guidance scale of image embedding")
    parser.add_argument('--s2s', type=int, nargs='+', default=[2], help="guidance Scale of input image")
    parser.add_argument('--s3s', type=int, nargs='+', default=[4], help="guidance Scale of text prompt")
    parser.add_argument('--inference_steps', type=int, default=20, help="Number of inference steps")
    parser.add_argument('--strength', type=float, default=1.0, help="Strength of the image modification")

    args = parser.parse_args()

    # fix random seed 
    torch.manual_seed(3407)
    np.random.seed(3407)
    random.seed(3407)

    # Load UNet and Pipeline
    unet = load_unet(args.model_dir)
    pipe = load_pipeline(args, unet)

    # Load adapter
    pipe.adapter = load_adapter(pipe, args.model_dir)
    
    # Set the scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # Process the image
    process_image(args, pipe, args.seeds)

if __name__ == "__main__":
    main()
