# Copyright (c) 2023 The HuggingFace Team (diffusers)
# Copyright (c) 2023 Johanna Karras (DreamPose)
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# This file has been modified by Bytedance Ltd. and/or its affiliates on October 24, 2024.

# Original file (diffusers) was released under Apache License 2.0, with the full license text
# available at https://github.com/huggingface/diffusers/blob/main/LICENSE.

# Original file (DreamPose) was released under MIT License, with the full license text
# available at https://github.com/johannakarras/DreamPose/blob/main/LICENSE.

import inspect
from typing import Callable, List, Optional, Union
from einops import rearrange

import numpy as np
import torch, torchvision
import torch.nn.functional as F
from  torch.cuda.amp import autocast
from torchvision import transforms
from torchvision.utils import make_grid
from typing import Any, Callable, Dict, List, Optional, Union

import PIL
from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.configuration_utils import FrozenDict
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import DiffusionPipeline
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils.torch_utils import randn_tensor

from diffusers.utils import PIL_INTERPOLATION, deprecate, logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from models.models import get_unet, Embedding_Adapter

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


class StableDiffusionImg2ImgPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-guided image to image generation using Stable Diffusion.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.__init__
    def __init__(
        self,
        #adapter: Embedding_Adapter,
        vae: AutoencoderKL,
        image_encoder: CLIPVisionModel,
        clip_processor: CLIPProcessor,
        unet: UNet2DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        safety_checker: None,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = False,
        stochastic_sampling: bool = False,


    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.adapter = Embedding_Adapter().cuda()

        self.register_modules(
            #adapter=self.adapter,
            vae=vae,
            image_encoder=image_encoder,
            clip_processor=clip_processor,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        # self.processor = self.clip_processor
        # self.clip_encoder = self.image_encoder.cuda()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

        self.vae = self.vae.cuda()
        self.unet = self.unet.cuda()
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

        self.fixed_noise = None
        self.stochastic_sampling = stochastic_sampling

        print("Stochastic Sampling: ", self.stochastic_sampling)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_sequential_cpu_offload
    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.image_encoder, self.clip_processor, self.vae, self.adapter]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            # TODO(Patrick) - there is currently a bug with cpu offload of nn.Parameter in accelerate
            # fix by only offloading self.safety_checker for now
            cpu_offload(self.safety_checker.vision_model, device)

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,

    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not self.use_peft_backend:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)
        # print(prompt_embeds.shape)
        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if isinstance(self, LoraLoaderMixin) and self.use_peft_backend:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder)

        return prompt_embeds, negative_prompt_embeds

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        **kwargs,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_image(self, image, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, dtype):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
            image (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        batch_size = len(image) if isinstance(image, list) else 1
        #print("Batch size = ", batch_size)

        if isinstance(image, list):
            uncond_image = [torch.zeros((image[0].size[0], image[0].size[1], 3)) for _ in range(batch_size)]
        else:
            image = [image]
            uncond_image = [torch.zeros((image[0].size[0], image[0].size[1], 3))]

        # clip encoder
        inputs = self.processor(images=image, return_tensors="pt")
        clip_image_embeddings = self.clip_encoder(**inputs).last_hidden_state.cuda().to(dtype)

        uncond_inputs = self.processor(images=uncond_image, return_tensors="pt")
        clip_uncond_image_embeddings = self.clip_encoder(**uncond_inputs).last_hidden_state.cuda().to(dtype)

        # vae encoder
        image_tensor = torch.tensor([np.array(im).transpose((2,0,1)) for im in image]).cuda().to(dtype)
        image_tensor = image_tensor / 255.0
        image_tensor = (image_tensor - 0.5 ) / 0.5 

        # print(image_tensor.dtype )
        # exit()

        # print(self.vae)
        # print(image_tensor.max(), image_tensor.min())

        vae_image_embeddings = self.vae.encode(image_tensor).latent_dist.sample() * 0.18215


        uncond_image_tensor = torch.tensor([np.array(im).transpose((2,0,1)) for im in uncond_image]).cuda().to(dtype)
        uncond_image_tensor = uncond_image_tensor / 255.0
        uncond_image_tensor = (uncond_image_tensor - 0.5 ) / 0.5
        # print(uncond_image_tensor.max(), uncond_image_tensor.min())

        vae_uncond_image_embeddings = self.vae.encode(uncond_image_tensor).latent_dist.sample() * 0.18215

        # print(vae_image_embeddings.dtype, clip_image_embeddings.dtype, )
        # print(self.adapter.dtype)
        
        # adapt embeddings
        image_embeddings = self.adapter(clip_image_embeddings, vae_image_embeddings)
        uncond_image_embeddings = self.adapter(clip_uncond_image_embeddings, vae_uncond_image_embeddings)

        #print(image_embeddings.shape)
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings  = image_embeddings.repeat(1, num_images_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        bs_embed, seq_len, _ = uncond_image_embeddings .shape
        uncond_image_embeddings  = uncond_image_embeddings.repeat(1, num_images_per_prompt, 1)
        uncond_image_embeddings = uncond_image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            image_embeddings = torch.cat([uncond_image_embeddings, image_embeddings, image_embeddings, image_embeddings])

        image_embeddings = image_embeddings.to(dtype=dtype)
        vae_image_embeddings = vae_image_embeddings.to(dtype=dtype)
        return image_embeddings, vae_image_embeddings

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        with autocast():
            latents = 1 / 0.18215 * latents
            image = self.vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
            image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
            return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, strength, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [1.0, 1.0] but is {strength}")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        with autocast():
            image = image.to(device=device, dtype=dtype).cuda()
            init_latent_dist = self.vae.encode(image).latent_dist
            init_latents = init_latent_dist.sample(generator=generator)
            init_latents = 0.18215 * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt * num_images_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents] * num_images_per_prompt, dim=0)

        # add noise to latents using the timesteps
        if self.fixed_noise is None:
            #print("Latents Shape = ", init_latents.shape, init_latents[0].shape, image.shape[0])
            single_fixed_noise = torch.randn(init_latents[0].shape, generator=generator, device=device, dtype=dtype)
            self.fixed_noise = single_fixed_noise.repeat(image.shape[0], 1, 1, 1)#torch.tensor([single_fixed_noise for _ in range(image.shape[0])])
        noise = self.fixed_noise

        # get latents
        init_latents = self.scheduler.add_noise(init_latents.cuda(), noise.cuda(), timestep)
        latents = init_latents

        return latents



    def prepare_randn_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents


    @torch.no_grad()
    def generate_mask(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        target_prompt: Optional[Union[str, List[str]]] = None,
        target_negative_prompt: Optional[Union[str, List[str]]] = None,
        target_prompt_embeds: Optional[torch.FloatTensor] = None,
        target_negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        source_prompt: Optional[Union[str, List[str]]] = None,
        source_negative_prompt: Optional[Union[str, List[str]]] = None,
        source_prompt_embeds: Optional[torch.FloatTensor] = None,
        source_negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        num_maps_per_mask: Optional[int] = 10,
        mask_encode_strength: Optional[float] = 0.5,
        mask_thresholding_ratio: Optional[float] = 3.0,
        num_inference_steps: int = 50,
        s1: float = 1.0, # strength of input pose
        s2: float = 1.0, # strength of input image
        s3: float = 1.0, # strength of input text
        combine_type: str = None,
        num_images_per_prompt: int = 1,

        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "np",
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        no_image_embedding: bool = False,
        no_concat: bool = False,
    ):
        r"""
        Generate a latent mask given a mask prompt, a target prompt, and an image.

        Args:
            image (`PIL.Image.Image`):
                `Image` or tensor representing an image batch to be used for computing the mask.
            target_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide semantic mask generation. If not defined, you need to pass
                `prompt_embeds`.
            target_negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            target_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            target_negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            source_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide semantic mask generation using DiffEdit. If not defined, you need to
                pass `source_prompt_embeds` or `source_image` instead.
            source_negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide semantic mask generation away from using DiffEdit. If not defined, you
                need to pass `source_negative_prompt_embeds` or `source_image` instead.
            source_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings to guide the semantic mask generation. Can be used to easily tweak text
                inputs (prompt weighting). If not provided, text embeddings are generated from `source_prompt` input
                argument.
            source_negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings to negatively guide the semantic mask generation. Can be used to easily
                tweak text inputs (prompt weighting). If not provided, text embeddings are generated from
                `source_negative_prompt` input argument.
            num_maps_per_mask (`int`, *optional*, defaults to 10):
                The number of noise maps sampled to generate the semantic mask using DiffEdit.
            mask_encode_strength (`float`, *optional*, defaults to 0.5):
                The strength of the noise maps sampled to generate the semantic mask using DiffEdit. Must be between 0
                and 1.
            mask_thresholding_ratio (`float`, *optional*, defaults to 3.0):
                The maximum multiple of the mean absolute difference used to clamp the semantic guidance map before
                mask binarization.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the
                [`~models.attention_processor.AttnProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

        Examples:

        Returns:
            `List[PIL.Image.Image]` or `np.array`:
                When returning a `List[PIL.Image.Image]`, the list consists of a batch of single-channel binary images
                with dimensions `(height // self.vae_scale_factor, width // self.vae_scale_factor)`. If it's
                `np.array`, the shape is `(batch_size, height // self.vae_scale_factor, width //
                self.vae_scale_factor)`.
        """

        if (num_maps_per_mask is None) or (
            num_maps_per_mask is not None and (not isinstance(num_maps_per_mask, int) or num_maps_per_mask <= 0)
        ):
            raise ValueError(
                f"`num_maps_per_mask` has to be a positive integer but is {num_maps_per_mask} of type"
                f" {type(num_maps_per_mask)}."
            )

        if mask_thresholding_ratio is None or mask_thresholding_ratio <= 0:
            raise ValueError(
                f"`mask_thresholding_ratio` has to be positive but is {mask_thresholding_ratio} of type"
                f" {type(mask_thresholding_ratio)}."
            )

        # 2. Define call parameters
        if target_prompt is not None and isinstance(target_prompt, str):
            batch_size = 1
        elif target_prompt is not None and isinstance(target_prompt, list):
            batch_size = len(target_prompt)
        else:
            batch_size = target_prompt_embeds.shape[0]
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = s1 > 1.0 or s2 > 1.0 or s3 > 1.0

        # 3. Encode input prompts
        (cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None)
        target_prompt_embeds, target_negative_prompt_embeds = self.encode_prompt(
            target_prompt,
            device,
            num_maps_per_mask,
            do_classifier_free_guidance,
            target_negative_prompt,
            prompt_embeds=target_prompt_embeds,
            negative_prompt_embeds=target_negative_prompt_embeds,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            target_prompt_embeds = torch.cat([target_negative_prompt_embeds, target_prompt_embeds, target_negative_prompt_embeds, target_negative_prompt_embeds])



        source_prompt_embeds, source_negative_prompt_embeds = self.encode_prompt(
            source_prompt,
            device,
            num_maps_per_mask,
            do_classifier_free_guidance,
            source_negative_prompt,
            prompt_embeds=source_prompt_embeds,
            negative_prompt_embeds=source_negative_prompt_embeds,
        )
        if do_classifier_free_guidance:
            source_prompt_embeds = torch.cat([source_negative_prompt_embeds, source_prompt_embeds, source_negative_prompt_embeds, source_negative_prompt_embeds])
    

        # 4. Encode input image: [unconditional, condional, conditional]
        embeddings, vae_image_embeddings = self._encode_image(
            image, device, num_images_per_prompt, do_classifier_free_guidance, '', dtype=self.unet.dtype
        )


        image = preprocess(image)
        image = torch.tensor(image).to( dtype=embeddings.dtype)


        # 5. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, _ = self.get_timesteps(num_inference_steps, mask_encode_strength, device)
        encode_timestep = timesteps[0]

        # 6. Prepare image latents and add noise with specified strength
        latents = self.prepare_latents(
                image, encode_timestep, batch_size * num_maps_per_mask, num_images_per_prompt, embeddings.dtype, device, generator
            ).to( dtype=embeddings.dtype)



        latent_model_input = torch.cat([latents] * (8 if do_classifier_free_guidance else 2))

       

        latent_model_input = self.scheduler.scale_model_input(latent_model_input, encode_timestep)

        if do_classifier_free_guidance:         
            concat_input = torch.cat([torch.zeros(vae_image_embeddings.shape).cuda(), vae_image_embeddings, torch.zeros(vae_image_embeddings.shape).cuda(), vae_image_embeddings]) 
        else:
            concat_input = vae_image_embeddings
 

        total_batch = latent_model_input.shape[0]

        concat_input = concat_input.repeat(total_batch // concat_input.shape[0], 1, 1, 1)
        
        if no_concat:
            latent_model_input = latent_model_input
        else:
            latent_model_input = torch.cat((latent_model_input.cuda(), concat_input), 1)

        if no_image_embedding:
            encoder_hidden_states1 = source_prompt_embeds
            encoder_hidden_states2 = target_prompt_embeds 
            encoder_hidden_states = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=0)

        else:
            embeddings = embeddings.repeat(source_prompt_embeds.shape[0] // embeddings.shape[0], 1, 1)
            if combine_type == 'add':
                encoder_hidden_states1 = source_prompt_embeds + embeddings
                encoder_hidden_states2 = target_prompt_embeds + embeddings
                encoder_hidden_states = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=0)


            elif combine_type == 'concat':
                encoder_hidden_states1 = torch.cat([source_prompt_embeds, embeddings], dim=1)
                encoder_hidden_states2 = torch.cat([target_prompt_embeds, embeddings], dim=1)
                encoder_hidden_states = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=0)

            elif combine_type == 'img_only':
                encoder_hidden_states = embeddings

        noise_pred = self.unet(
            latent_model_input,
            encode_timestep,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample

        if do_classifier_free_guidance:
            noise_pred_uncond_src, noise_pred_all_src, noise_pred_img_p_src,  noise_pred_img_p_concat_src,noise_pred_uncond_tgt, noise_pred_all_tgt, noise_pred_img_p_tgt,  noise_pred_img_p_concat_tgt = noise_pred.chunk(8) 

            noise_pred_source = noise_pred_uncond_src + \
                s1 * (noise_pred_img_p_src - noise_pred_uncond_src) + \
                s2 * (noise_pred_img_p_concat_src - noise_pred_img_p_src) + \
                s3 * (noise_pred_all_src - noise_pred_img_p_concat_src)# image embed. input channel, text embedding


            noise_pred_target = noise_pred_uncond_tgt + \
                s1 * (noise_pred_img_p_tgt - noise_pred_uncond_tgt) + \
                s2 * (noise_pred_img_p_concat_tgt - noise_pred_img_p_tgt) + \
                s3 * (noise_pred_all_tgt - noise_pred_img_p_concat_tgt)# image embed. input channel, text embedding



        else:
            noise_pred_source, noise_pred_target = noise_pred.chunk(2)

        # 8. Compute the mask from the absolute difference of predicted noise residuals
        # TODO: Consider smoothing mask guidance map
        mask_guidance_map = (
            torch.abs(noise_pred_target - noise_pred_source)
            .reshape(batch_size, num_maps_per_mask, *noise_pred_target.shape[-3:])
            .mean([1, 2])
        )
        # print(mask_guidance_map.mean())
        clamp_magnitude = mask_guidance_map.mean() * mask_thresholding_ratio
        semantic_mask_image = mask_guidance_map.clamp(0, clamp_magnitude) / clamp_magnitude
        # semantic_mask_image = torch.where(semantic_mask_image <= 0.5, 0, 1)
        mask_image = semantic_mask_image.cpu().numpy()

        # 9. Convert to Numpy array or PIL.
        if output_type == "pil":
            # mask_image = self.image_processor.numpy_to_pil(mask_image)

            from PIL import Image

            mask_image = np.transpose(mask_image, (1,2,0))
            # to rgb 
            mask_image = np.concatenate([mask_image, mask_image, mask_image], axis=2)

            mask_image = Image.fromarray(np.uint8(mask_image * 255))


        return mask_image



    # @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]]=None ,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        strength: float = 1.0,
        s1: float = 1.0, # strength of input pose
        s2: float = 1.0, # strength of input image
        s3: float = 1.0, # strength of input text
        combine_type: str = None, 
        fast_latents: List[int] = None,
        latents: torch.FloatTensor = None, 
        num_inference_steps: Optional[int] = 100,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        return_latents: bool = False,
        use_randn_latent: bool = False,
        rec_latents: torch.FloatTensor = None, 
        mask: torch.FloatTensor = None,
        prompt_embeds: torch.FloatTensor = None,
        negative_prompt_embeds: torch.FloatTensor = None,
        source_prompt: torch.FloatTensor = None,
        source_prompt_embeds: torch.FloatTensor = None,
        source_negative_prompt_embeds: torch.FloatTensor = None,
        no_image_embedding: bool = False,
        no_concat: bool = False,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter will be modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        # 1. Check inputs
        # self.check_inputs(prompt, strength, callback_steps)

        if prompt_embeds is not None and prompt is not None:
            raise ValueError("Only one of `prompt` or `prompt_embeds` should be provided.")

        # 2. Set adapter
        # if adapter is not None:
        #     print("Setting adapter")
        #     self.adapter = adapter

        assert self.adapter is not None

        # 3. Define call parameters
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
            
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = s1 > 1.0 or s2 > 1.0

        # 4. Encode input image: [unconditional, condional, conditional]
        embeddings, vae_image_embeddings = self._encode_image(
            image, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, dtype=self.unet.dtype
        )


        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,

        )


        if source_prompt is not None:
            source_prompt_embeds, source_negative_prompt_embeds = self.encode_prompt(
            source_prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=source_prompt_embeds,
            negative_prompt_embeds=source_negative_prompt_embeds,

            )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds, negative_prompt_embeds, negative_prompt_embeds])


        # print(image)

        # 5. Preprocess image
        image = preprocess(image)
        image = torch.tensor(image).to( dtype=embeddings.dtype)
        # print(image.max(), image.min())
        # 6. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 7. Prepare latent variables
        if latents is None:
            if not use_randn_latent:
                latents = self.prepare_latents(
                    image, latent_timestep, batch_size, num_images_per_prompt, embeddings.dtype, device, generator
                ).to( dtype=embeddings.dtype)
            else:
                # print('random')
                latents = self.prepare_randn_latents(
                    batch_size * num_images_per_prompt,
                    self.unet.config.in_channels,
                    512,
                    512,
                    embeddings.dtype,
                    device,
                    generator,
                )
        else:
            latents = latents.to( dtype=embeddings.dtype).cuda()
            # print('use predefined init latents ')
        # latents = torch.randn(latents.shape, generator=generator, device=device, dtype=embeddings.dtype)
        # print('done')


        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 10. Denoising loop
        copy_latents = latents.clone()
        latents = copy_latents.clone()
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        selected_latents = []

        if fast_latents is not None:
            # get max element from list
            max_fast_latent = max(fast_latents)
            
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                t = t.cuda()    
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 4) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)



                # Add image to noisy latents
                _, _, h, w = latent_model_input.shape


                if do_classifier_free_guidance: 
                    concat_input = torch.cat([torch.zeros(vae_image_embeddings.shape).cuda(), vae_image_embeddings, torch.zeros(vae_image_embeddings.shape).cuda(), vae_image_embeddings]) 
                else:
                    concat_input = vae_image_embeddings
                
                
                

                if no_concat:
                    latent_model_input = latent_model_input
                    # print("no concat")
                else:
                    latent_model_input = torch.cat((latent_model_input.cuda(), concat_input), 1)


    
                if no_image_embedding:
                    encoder_hidden_states = prompt_embeds
                    # print("no image embedding")

                else:
                    if combine_type == 'add':
                        encoder_hidden_states = prompt_embeds + embeddings

                    elif combine_type == 'concat':
                        encoder_hidden_states = torch.cat([prompt_embeds, embeddings], dim=1)

                    elif combine_type == 'img_only':
                        encoder_hidden_states = embeddings

                # print(encoder_hidden_states.shape)
                



                # exit()
                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states).sample

                if source_prompt is not None:
                    if combine_type == 'add':
                        src_encoder_hidden_states = source_prompt_embeds + embeddings

                    elif combine_type == 'concat':
                        src_encoder_hidden_states = torch.cat([source_prompt_embeds, embeddings], dim=1)

                    elif combine_type == 'img_only':
                        src_encoder_hidden_states = embeddings

                    src_noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=src_encoder_hidden_states).sample


                # perform guidance
                if do_classifier_free_guidance:
                    #print(f"s1={s1}, s2={s2}")
                    noise_pred_uncond, noise_pred_all, noise_pred_img_p,  noise_pred_img_p_concat = noise_pred.chunk(4)
                    noise_pred = noise_pred_uncond + \
                                    s1 * (noise_pred_img_p - noise_pred_uncond) + \
                                    s2 * (noise_pred_img_p_concat - noise_pred_img_p) + \
                                    s3 * (noise_pred_all - noise_pred_img_p_concat)# image embed. input channel, text embedding

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred.cuda(), t, latents.cuda(), **extra_step_kwargs).prev_sample

                if mask is not None:
                    rec_latent = rec_latents[i:i+1]
                    latents = latents * mask + rec_latent * (1 - mask)


                if fast_latents is not None:
                    if i in fast_latents:
                        selected_latents.append(latents.clone())

                    if i == max_fast_latent:
                        
                        print(f"End fast latents {i}")
                        return selected_latents

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        # print(latents.shape)
        # 11. Post-processing
        latents = latents[:,:4, :, :].cuda() #.float()
        image = self.decode_latents(latents.clone().detach())


        # 13. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, False)

        if return_latents:
            return image, latents
        else:
            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=False)

