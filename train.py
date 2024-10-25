# Copyright (c) 2023 The HuggingFace Team (diffusers)
# Copyright (c) 2023 Johanna Karras (DreamPose)
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# This file has been modified by Bytedance Ltd. and/or its affiliates on October 24, 2024.

# Original file (diffusers) was released under Apache License 2.0, with the full license text
# available at https://github.com/huggingface/diffusers/blob/main/LICENSE.

# Original file (DreamPose) was released under MIT License, with the full license text
# available at https://github.com/johannakarras/DreamPose/blob/main/LICENSE.

import argparse
import itertools
import math
import os
import random
from pathlib import Path
from typing import Optional
from collections import OrderedDict
import matplotlib.pyplot as plt
import json
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import cv2
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTokenizer, CLIPProcessor, CLIPVisionModel
from diffusers import UniPCMultistepScheduler
from transformers import AutoTokenizer, PretrainedConfig

import sys
logger = get_logger(__name__)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.parse_args import parse_args
from dataset.dataset import MCDMDataset
from models.models import get_unet, Embedding_Adapter


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

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



def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds



def main(args):

    combine_type = args.combine_type
    dropout_rate = args.dropout_rate
    rec_only = args.rec_only
    reconst_prob = args.reconst_prob
    no_concat = args.no_concat
    no_image_embedding = args.no_image_embedding
    random_init = args.random_init

    instance_data_name = args.instance_data_dir.split("/")[-2]

    if random_init:
        model_name = 'random'

    else:
        model_name = args.pretrained_model_name_or_path.split("/")[-1]


    args.output_dir = f'{args.output_dir}/{combine_type}_drop{dropout_rate}'

    if rec_only:
        args.output_dir = f'{args.output_dir}_rec_only'

    if reconst_prob > 0:
        args.output_dir = f'{args.output_dir}_reconst{reconst_prob}'

    if no_concat:
        args.output_dir = f'{args.output_dir}_no_concat'

    if no_image_embedding:
        args.output_dir = f'{args.output_dir}_no_image_embedding'


    logging_dir = Path(args.output_dir, args.logging_dir)




    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=logging_dir,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load CLIP Image Encoder
    clip_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
    clip_encoder.requires_grad_(False)

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # Load models and create wrapper for stable diffusion
    vae = AutoencoderKL.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="vae",
            )

    # Load pretrained UNet layers
    additional_channel = 0 if no_concat else 4
    unet = get_unet(args.pretrained_model_name_or_path, args.revision, additional_channel=additional_channel, resolution=args.resolution, random_init=random_init)


    tokenizer = AutoTokenizer.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=args.revision,
                use_fast=False,
            )


    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    # Embedding adapter
    if combine_type == 'add':
        adapter = Embedding_Adapter(output_nc=77)

    elif combine_type == 'concat':
        adapter = Embedding_Adapter(output_nc=50)




    adapter.requires_grad_(True)
    unet.requires_grad_(True)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)


    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        adapter.enable_gradient_checkpointing()
        

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )


    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )


    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW


    params_to_optimize = (
        itertools.chain(unet.parameters(), adapter.parameters(),)
    )

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )

    train_dataset = MCDMDataset(
        instance_data_root=args.instance_data_dir,
        size=args.resolution,
        center_crop=args.center_crop,
        tokenizer=tokenizer,
        tokenizer_max_length=args.tokenizer_max_length,
        dropout_rate=args.dropout_rate,
        rec_only=args.rec_only,
        reconst_prob=reconst_prob,
    )

    def collate_fn(examples):
        input_image_emb = [example["input_image_emb"] for example in examples]
        input_image_concat = [example["input_image_concat"] for example in examples]
        target_image = [example["target_image"] for example in examples]
        input_image_PIL = [example["input_image_PIL"] for example in examples]
        prompt = [example["prompt"] for example in examples]

        
        input_ids = [example["instance_prompt_ids"] for example in examples]
        has_attention_mask = "instance_attention_mask" in examples[0]

        if has_attention_mask:
            attention_mask = [example["instance_attention_mask"] for example in examples]

        input_image_emb = torch.stack(input_image_emb, 0)
        input_image_concat = torch.stack(input_image_concat, 0)
        target_image = torch.stack(target_image, 0)
        input_ids = torch.cat(input_ids, dim=0)

        input_image_emb = input_image_emb.to(memory_format=torch.contiguous_format).float()
        input_image_concat = input_image_concat.to(memory_format=torch.contiguous_format).float()


        batch = {
            "input_image_emb": input_image_emb,
            'input_image_concat': input_image_concat,
            "target_image": target_image,
            'input_image_PIL': input_image_PIL,
            "prompt": prompt,
            "input_ids": input_ids,

        }

        if has_attention_mask:
            attention_mask = torch.cat(attention_mask, dim=0)
            batch["attention_mask"] = attention_mask

        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    unet, adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, adapter, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the image_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("editing", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")



    global_step = 0
    if args.resume_from_checkpoint:

        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        print(f'loading from {args.output_dir}/{path}')

        
        adapter_chkpt = f'{args.output_dir}/{path}/adapter.pth'

        adapter_state_dict = torch.load(adapter_chkpt)
        new_state_dict = OrderedDict()
        for k, v in adapter_state_dict.items():
            name = k[7:] if k[:7] == 'module' else k 
            new_state_dict[name] = v
        adapter.load_state_dict(new_state_dict)
        adapter = adapter.cuda()


        unet_chkpt = f'{args.output_dir}/{path}/unet.pth'
        unet_state_dict = torch.load(unet_chkpt)
        new_state_dict = OrderedDict()
        for k, v in unet_state_dict.items():
            name = k[7:] if k[:7] == 'module' else k 
            new_state_dict[name] = v
        unet.load_state_dict(new_state_dict)
        unet = unet.cuda()


        global_step = int(path.split("-")[1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)


    latest_chkpt_step = 0
    for epoch in range(args.epoch, args.num_train_epochs):
        unet.train()
        adapter.train()
        for step, batch in enumerate(train_dataloader):
        
            
            if args.resume_from_checkpoint and epoch <= first_epoch:
                print(epoch, first_epoch, step, resume_step)
                if epoch == first_epoch:
                    if step < resume_step:
                        if step % args.gradient_accumulation_steps == 0:
                            progress_bar.update(1)
                        continue

                else:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue
                
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["target_image"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215

                latents_input = vae.encode(batch["input_image_concat"].to(dtype=weight_dtype)).latent_dist.sample()
                latents_input = latents_input * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Concatenate pose with noise
                _, _, h, w = noisy_latents.shape

                if not no_concat:
                    noisy_latents = torch.cat((noisy_latents, latents_input), 1)


                inputs = clip_processor(images=list(batch['input_image_PIL']), return_tensors="pt")
                inputs = {k: v.to(latents.device) for k, v in inputs.items()}
                clip_hidden_states =  clip_encoder(**inputs).last_hidden_state.to(latents.device)


                # Get VAE embeddings
                vae_hidden_states = vae.encode(batch["input_image_emb"].to(dtype=weight_dtype)).latent_dist.sample()
                vae_hidden_states = vae_hidden_states * 0.18215
    

                encoder_hidden_states_img = adapter(clip_hidden_states, vae_hidden_states)


                encoder_hidden_states_text = encode_prompt(
                        text_encoder,
                        batch["input_ids"],
                        batch["attention_mask"],
                        text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                    )
            



                
                if no_image_embedding:
                    encoder_hidden_states = encoder_hidden_states_text

                else:
                    if combine_type == 'add':
                        encoder_hidden_states = encoder_hidden_states_text + encoder_hidden_states_img

                    elif combine_type == 'concat':
                        
                        encoder_hidden_states = torch.cat([encoder_hidden_states_text, encoder_hidden_states_img], dim=1)

                    elif combine_type == 'img_only':
                        encoder_hidden_states = encoder_hidden_states_img
                # print(encoder_hidden_states_text.shape, encoder_hidden_states_img.shape, encoder_hidden_states.shape)

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
             
                    params_to_clip = (
                        itertools.chain(unet.parameters())
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1



                            

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
            # save model
            if accelerator.is_main_process and global_step % args.save_interval == 0:
                print(f'save to ', os.path.join(args.output_dir, f'checkpoint-{global_step}'))
                os.makedirs(os.path.join(args.output_dir, f'checkpoint-{global_step}'), exist_ok=True)    
                model_path = args.output_dir+f'/checkpoint-{global_step}/unet.pth'
                torch.save(accelerator.unwrap_model(unet).state_dict(), model_path)
                adapter_path = args.output_dir+f'/checkpoint-{global_step}/adapter.pth'
                torch.save(accelerator.unwrap_model(adapter).state_dict(), adapter_path)

  
        accelerator.wait_for_everyone()

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)