# Copyright (c) 2023 Johanna Karras (DreamPose)
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# This file has been modified by Bytedance Ltd. and/or its affiliates on October 24, 2024.

# Original file (DreamPose) was released under MIT License, with the full license text
# available at https://github.com/johannakarras/DreamPose/blob/main/LICENSE.


from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
import os, glob
import json




def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs






class MCDMDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        size=512,
        center_crop=False,
        train=True,
        p_jitter=0.9,
        class_num=50, 
        tokenizer=None, 
        tokenizer_max_length=None, 
        dropout_rate=None,
        rec_only=None, 
        reconst_prob=None,
        
    ):
        self.size = (512, 512)
        self.center_crop = center_crop
        self.train = train
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.dropout_rate = dropout_rate
        self.rec_only = rec_only
        self.reconst_prob = reconst_prob


        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")


        self.data_list = []
        with open(f'{instance_data_root}/prompt.json', 'rt') as f:
            for line in f:
                self.data_list.append(json.loads(line))

    
        self.num_instance_images = len(self.data_list)
        self._length = self.num_instance_images


        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                # transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )



    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}

        data = self.data_list[index % self.num_instance_images]

        source_filename = data['source']
        target_filename = data['target']
        prompt = data['prompt']

        if self.rec_only:
            prompt = data['source_prompt']
            target_filename = data['source']

        p = np.random.random()
        if p <= self.reconst_prob:
            prompt = data['source_prompt']
            target_filename = data['source']


        ref_image = Image.open(f'{self.instance_data_root}/' + source_filename)
        ref_image_PIL = ref_image.copy().convert("RGB").resize((224,224), Image.BILINEAR)

        if not ref_image.mode == "RGB":
            ref_image = ref_image.convert("RGB")

        instance_image = Image.open(f'{self.instance_data_root}/' + target_filename)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        input_image_concat = self.image_transforms(ref_image)
        input_image_emb = input_image_concat.clone()
        target_image = self.image_transforms(instance_image)

        

        # Dropout
        p = np.random.random()
        if p <= self.dropout_rate / 4: # dropout pose
            input_image_concat = torch.zeros(input_image_concat.shape)
        elif p <= 2*self.dropout_rate / 4: # dropout image
            input_image_emb = torch.zeros(input_image_emb.shape)
            ref_image_PIL = Image.new('RGB', (224,224), (0,0,0))
        elif p <= 3*self.dropout_rate / 4: # dropout image
            prompt = ''
        elif p <= self.dropout_rate: # dropout image and pose
            input_image_emb = torch.zeros(input_image_emb.shape)
            input_image_concat = torch.zeros(input_image_concat.shape)
            prompt = ''
            ref_image_PIL = Image.new('RGB', (224,224), (0,0,0))
        


        text_inputs = tokenize_prompt(
            self.tokenizer, prompt, tokenizer_max_length=self.tokenizer_max_length
        )
        example["instance_prompt_ids"] = text_inputs.input_ids
        example["instance_attention_mask"] = text_inputs.attention_mask

        example["input_image_emb"] = input_image_emb
        example["input_image_concat"] = input_image_concat
        example["input_image_PIL"] = ref_image_PIL
        example["target_image"] = target_image
        example["prompt"] = prompt
        

        return example
