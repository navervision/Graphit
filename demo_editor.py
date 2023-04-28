"""
Graphit
Copyright (c) 2023-present NAVER Corp.
Apache-2.0
"""
import os
import numpy as np
import base64
import requests
from io import BytesIO
import json
import time
import math
import argparse

import torch
import torch.nn.functional as F
import gradio as gr

import types
from typing import Union, List, Optional, Callable
import diffusers
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from diffusers.models import AutoencoderKL
from transformers import CLIPTextModel

import datasets

from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, pil_to_tensor

import PIL
from PIL import Image, ImageOps

import compodiff
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from transparent_background import Remover
from huggingface_hub import hf_hub_url, cached_download
from RealESRGAN import RealESRGAN
import einops
import cv2
from skimage import segmentation, color, graph
import random


def preprocess(image, mode):
    image = np.array(image)[None, :].astype(np.float32) / 255.0
    image = image
    image = image.transpose(0, 3, 1, 2)
    image = 2.0 * image - 1.0
    if mode == 'scr2i':
        image[image > 0.0] = 0.0
    image = torch.from_numpy(image)
    return image


class GraphitPipeline(StableDiffusionInstructPix2PixPipeline):
    '''
    override:
        /opt/conda/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
    '''
    def prepare_image_latents(
        self, image, mask, batch_size, num_images_per_prompt, dtype, device, do_classifier_free_guidance, generator=None
    ):
        if not isinstance(image, (torch.Tensor, Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            image_latents = [self.vae.encode(image[i : i + 1]).latent_dist.mode() for i in range(batch_size)]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = self.vae.encode(image).latent_dist.mode()

        mask = torch.nn.functional.interpolate(
                mask, #.unsqueeze(0).unsqueeze(0),
                size=(image_latents.shape[-2], image_latents.shape[-1]),
                mode='bicubic',
                align_corners=False,
                )

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # expand image_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {image_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            #deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            mask = torch.cat([mask] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)
        image_latents *= 0.18215
        if do_classifier_free_guidance:
            uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([image_latents, image_latents], dim=0)
            mask = torch.cat([mask, mask], dim=0)
            image_latents = torch.cat([image_latents, mask], dim=1)

        return image_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask: Union[torch.FloatTensor, PIL.Image.Image] = None,
        depth_map: Union[torch.FloatTensor, PIL.Image.Image] = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 3.5,
        use_depth_map_as_input: bool = False,
        apply_mask_to_input: bool = True,
        mode: str = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        image_cond_embeds: Optional[torch.FloatTensor] = None,
        negative_image_cond_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        # 0. Check inputs
        self.check_inputs(prompt, callback_steps)

        if image is None:
            raise ValueError("`image` input cannot be undefined.")

        # 1. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = True#guidance_scale >= 1.0 and image_guidance_scale >= 1.0
        # check if scheduler is in sigmas space
        scheduler_is_in_sigma_space = hasattr(self.scheduler, "sigmas")

        # 2. Encode input prompt
        cond_embeds = torch.cat([image_cond_embeds, negative_image_cond_embeds])
        cond_embeds = einops.repeat(cond_embeds, 'b n d -> (b num) n d', num=num_images_per_prompt).to(torch.float16)
        prompt_embeds = cond_embeds

        # 3. Preprocess image
        image = preprocess(image, mode)

        if len(mask.shape) > 2:
            edge_map = mask[:,:,1:]
            edge_map = preprocess(edge_map, mode)
            mask = mask[:,:,0]
        else:
            edge_map = None
        mask = mask.unsqueeze(0).unsqueeze(0)
        if torch.sum(mask).item() == 0.0 and use_depth_map_as_input:
            image = depth_map
        if edge_map is None:
            if apply_mask_to_input:
                image = image * (1 - mask)
        else:
            image = image * (1 - mask) + edge_map * mask
        height, width = image.shape[-2:]

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare Image latents
        image_latents = self.prepare_image_latents(
            image,
            mask,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            do_classifier_free_guidance,
            generator,
        )

        if mode == 't2i':
            image_latents = torch.zeros_like(image_latents)

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Check that shapes of latents and image match the UNet channels
        num_channels_image = image_latents.shape[1]
        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents+num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance.
                # The latents are expanded 3 times because for pix2pix the guidance\
                # is applied for both the text and the input image.
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # concat latents, image_latents in the channel dimension
                scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)

                # predict the noise residual
                noise_pred = self.unet(scaled_latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

                # Hack:
                # For karras style schedulers the model does classifer free guidance using the
                # predicted_original_sample instead of the noise_pred. So we need to compute the
                # predicted_original_sample here if we are using a karras style scheduler.
                if scheduler_is_in_sigma_space:
                    step_index = (self.scheduler.timesteps == t).nonzero().item()
                    sigma = self.scheduler.sigmas[step_index]
                    noise_pred = latent_model_input - sigma * noise_pred

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_full, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = (
                        noise_pred_uncond
                        + guidance_scale * (noise_pred_full - noise_pred_uncond)
                    )

                # Hack:
                # For karras style schedulers the model does classifer free guidance using the
                # predicted_original_sample instead of the noise_pred. But the scheduler.step function
                # expects the noise_pred and computes the predicted_original_sample internally. So we
                # need to overwrite the noise_pred here such that the value of the computed
                # predicted_original_sample is correct.
                if scheduler_is_in_sigma_space:
                    noise_pred = (noise_pred - latents) / (-sigma)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 10. Post-processing
        image = self.decode_latents(latents)

        # 11. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # 12. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


class CustomRealESRGAN(RealESRGAN):
    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def predict(self, pil_lr_image_list):
        device = self.device
        # batchfy
        batch_lr_images = (torch.stack([pil_to_tensor(pil_lr_image) for pil_lr_image in pil_lr_image_list]).float() / 255).to(device)
        batch_outputs = self.model(batch_lr_images).clamp_(0, 1)

        # to pil images
        return [to_pil_image(output) for output in batch_outputs]


def build_models(args):
    # Load scheduler, tokenizer and models.

    model_path = 'navervision/Graphit-SD'
    unet = UNet2DConditionModel.from_pretrained(
        model_path, torch_dtype=torch.float16,
    )

    vae_name = 'stabilityai/sd-vae-ft-ema'
    vae = AutoencoderKL.from_pretrained(vae_name, torch_dtype=torch.float16)

    model_name = 'timbrooks/instruct-pix2pix'
    pipe = GraphitPipeline.from_pretrained(model_name, torch_dtype=torch.float16, safety_checker=None,
            unet = unet,
            vae = vae,
            )
    pipe = pipe.to('cuda:0')

    ## load CompoDiff
    compodiff_model, clip_model, clip_preprocess, clip_tokenizer = compodiff.build_model()
    compodiff_model, clip_model = compodiff_model.to('cuda:0'), clip_model.to('cuda:0')

    ## load third-party models
    model_name = 'Intel/dpt-large'
    depth_preprocess = DPTFeatureExtractor.from_pretrained(model_name)
    depth_predictor = DPTForDepthEstimation.from_pretrained(model_name, torch_dtype=torch.float16)
    depth_predictor = depth_predictor.to('cuda:0')

    if not os.path.exists('./third_party/remover_fast.pth'):
        model_file_url = hf_hub_url(repo_id='Geonmo/remover_fast', filename='remover_fast.pth')
        cached_download(model_file_url, cache_dir='./third_party', force_filename='remover_fast.pth')
    remover = Remover(fast=True, jit=False, device='cuda:0', ckpt='./third_party/remover_fast.pth')

    sr_model = CustomRealESRGAN('cuda:0', scale=2)
    sr_model.load_weights('./third_party/RealESRGAN_x2.pth', download=True)

    dataset = datasets.load_dataset("FredZhang7/stable-diffusion-prompts-2.47M")

    train = dataset["train"]
    prompts = train["text"]

    model_dict = {'pipe': pipe,
                  'compodiff': compodiff_model,
                  'clip_preprocess': clip_preprocess,
                  'clip_tokenizer': clip_tokenizer,
                  'clip_model': clip_model,
                  'depth_preprocess': depth_preprocess,
                  'depth_predictor': depth_predictor,
                  'remover': remover,
                  'sr_model': sr_model,
                  'prompt_candidates': prompts,
                  }
    return model_dict


def predict_compodiff(image, text_input, negative_text, cfg_image_scale, cfg_text_scale, mask, random_seed):
    text_token_dict = model_dict['clip_tokenizer'](text=text_input, return_tensors='pt', padding='max_length', truncation=True)
    text_tokens, text_attention_mask = text_token_dict['input_ids'].to('cuda:0'), text_token_dict['attention_mask'].to('cuda:0')

    negative_text_token_dict = model_dict['clip_tokenizer'](text=negative_text, return_tensors='pt', padding='max_length', truncation=True)
    negative_text_tokens, negative_text_attention_mask = negative_text_token_dict['input_ids'].to('cuda:0'), text_token_dict['attention_mask'].to('cuda:0')

    with torch.no_grad():
        if image is None:
            image_cond = torch.zeros([1,1,768]).to('cuda:0')
            mask = torch.tensor(np.zeros([64, 64], dtype='float32')).to('cuda:0').unsqueeze(0)
        else:
            image_source = image.resize((512, 512))
            image_source = model_dict['clip_preprocess'](image_source, return_tensors='pt')['pixel_values'].to('cuda:0')
            mask = mask.resize((512, 512))
            mask = model_dict['clip_preprocess'](mask, do_normalize=False, return_tensors='pt')['pixel_values']
            mask = mask[:,:1,:,:]
            mask = (mask > 0.5).float().to('cuda:0')
            image_source = image_source * (1 - mask)
            image_cond = model_dict['clip_model'].encode_images(image_source)
            mask = transforms.Resize([64, 64])(mask)[:,0,:,:]
            mask = (mask > 0.5).float()

        text_cond = model_dict['clip_model'].encode_texts(text_tokens, text_attention_mask)
        negative_text_cond = model_dict['clip_model'].encode_texts(negative_text_tokens, negative_text_attention_mask)

        sampled_image_features = model_dict['compodiff'].sample(image_cond, text_cond, negative_text_cond, mask, timesteps=25, cond_scale=(1.0 if image is None else 1.3, cfg_text_scale), num_samples_per_batch=4, random_seed=random_seed).unsqueeze(1)
    return sampled_image_features, image_cond


def generate_depth_map(image, height, width):
    depth_inputs = {k: v.to('cuda:0', dtype=torch.float16) for k, v in model_dict['depth_preprocess'](images=image, return_tensors='pt').items()}
    depth_map = model_dict['depth_predictor'](**depth_inputs).predicted_depth.unsqueeze(1)
    depth_min = torch.amin(depth_map, dim=[1,2,3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1,2,3], keepdim=True)
    depth_map = 2.0 * ((depth_map - depth_min) / (depth_max - depth_min)) - 1.0
    depth_map = torch.nn.functional.interpolate(
            depth_map,
            size=(height, width),
            mode='bicubic',
            align_corners=False,
            )
    return depth_map


def generate_color(image, compactness=30, n_segments=100, thresh=35, blur_kernel=3, blur_std=0):
    img = image # 0 ~ 255 uint8
    labels = segmentation.slic(img, compactness=compactness, n_segments=n_segments)#, start_label=1)
    g = graph.rag_mean_color(img, labels)
    labels2 = graph.cut_threshold(labels, g, thresh=thresh)
    out = color.label2rgb(labels2, img, kind='avg', bg_label=-1)
    return out


@torch.no_grad()
def generate(image_source, image_reference, text_input, negative_prompt, steps, random_seed, cfg_image_scale, cfg_text_scale, cfg_image_space_scale, cfg_image_reference_mix_weight, cfg_image_source_mix_weight, mask_scale, use_edge, t2i_height, t2i_width, do_sr, mode):
    text_input = text_input.lower()
    if negative_prompt == '':
        print('running without a negative prompt')
    # prepare an input image
    use_mask = False
    mask = None
    is_null_image_source = False
    if type(image_source) == dict:
        image_source, mask = image_source['image'], image_source['mask']
    elif image_source is None:
        image_source = Image.fromarray(np.zeros([t2i_height, t2i_width, 3]).astype('uint8'))
        is_null_image_source = True

    try:
        image_source = ImageOps.exif_transpose(image_source)
    except:
        pass

    width, height = image_source.size
    factor = 512 / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64

    image_source = org_image_source = ImageOps.fit(image_source, (width, height), method=Image.Resampling.LANCZOS)

    if mask is not None:
        mask_pil = mask = ImageOps.fit(mask, (width, height), method=Image.Resampling.LANCZOS)
        mask = ((torch.tensor(np.array(mask.convert('L'))).float() / 255.0) > 0.5).float()
        if torch.sum(mask).item() > 0.0:
            print('now using mask')
            use_mask = True
    else:
        mask = torch.zeros([height, width])
        mask_pil = to_pil_image(mask)

    use_depth_map_as_input = False
    if mode == 's2i' or mode == 'scr2i': # sketch to image
        image_source = mask
        image_source = einops.repeat(image_source, 'h w -> r h w', r=3)
        mask = image_source[0,:,:]
        image_source = org_image_source = to_pil_image(image_source)
        mask_pil = to_pil_image(mask)
        mask *= mask_scale
        use_mask = False
    elif mode == 'cs2i':
        mask = torch.tensor((np.array(image_source)[:,:,0] != 255)).float() * mask_scale
        mask_pil = Image.fromarray(((np.array(image_source)[:,:,0] != 255) * 255).astype('uint8'))
        use_mask = False #True
    elif mode == 'd2i': # depth to image
        use_depth_map_as_input = True
    elif mode == 'e2i': # edge to image
        image_source = einops.repeat(cv2.Canny(cv2.cvtColor(np.array(image_source)[:,:,::-1], cv2.COLOR_BGR2GRAY), threshold1=100, threshold2=200), 'h w -> h w r', r=3)
        image_source = Image.fromarray(image_source) #to_pil_image(image_source)
        org_image_source = image_source
    elif mode == 'inped':
        # mask = torch.Size([512, 512])
        mask_np = (einops.repeat(mask.numpy(), 'h w -> h w r', r=1) * 255).astype('uint8')
        gray = mask_np #cv2.cvtColor(mask_np, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        cv2.rectangle(mask_np, (x, y), (x+w, y+h), 255, -1)
        mask_np = mask_np.astype('float32') / 255
        if image_reference is not None:
            edge_reference = image_reference.resize((w, h))
            color_map = generate_color(np.array(edge_reference)).astype('float32')
            reference_map = (model_dict['remover'].process(edge_reference, type='map') > 16).astype('float32')
            edge_reference = einops.repeat(cv2.Canny(cv2.cvtColor(np.array(edge_reference)[:,:,::-1], cv2.COLOR_BGR2GRAY), threshold1=100, threshold2=200), 'h w -> h w r', r=3).astype('float32')
            edge_np = np.zeros_like(np.array(image_source)).astype('float32')
            if text_input != '':
                edge_np[y:y+h,x:x+w] = edge_reference * reference_map
            elif use_edge and mask_scale > 0.0:
                print('mode: color inped with with_edge')
                edge_np[y:y+h,x:x+w] = (255 - edge_reference) / 255 * color_map * reference_map + (1 - mask_scale) * edge_reference  / 255 * reference_map
            else:
                print('mode: color inped with no_edge')
                edge_np[y:y+h,x:x+w] = color_map * reference_map
            mask_np = np.zeros_like(np.array(image_source)).astype('float32')
            mask_np[y:y+h,x:x+w] = reference_map #edge_reference
            mask_np = mask_np[:,:,:1]
        else:
            edge_np = einops.repeat(cv2.Canny(cv2.cvtColor(np.array(image_source)[:,:,::-1], cv2.COLOR_BGR2GRAY), threshold1=100, threshold2=200), 'h w -> h w r', r=3).astype('float32')
        # concat edge to mask_np
        mask = torch.tensor(np.concatenate([mask_np, edge_np], axis=-1))
        mask_pil = to_pil_image(mask_np[:,:,0].astype('uint8') * 255)
        #mask_pil = to_pil_image((mask_np[:,:,0] * 255).astype('uint8'))

    with torch.no_grad():
        # do reference first
        if image_reference is not None:
            image_cond_reference = ImageOps.exif_transpose(image_reference)
            image_cond_reference = model_dict['clip_preprocess'](image_cond_reference, return_tensors='pt')['pixel_values'].to('cuda:0')
            image_cond_reference = model_dict['clip_model'].encode_images(image_cond_reference)
        else:
            image_cond_reference = torch.zeros([1, 1, 768]).to(torch.float16).to('cuda:0')

        # do source or knn
        image_cond_source = None
        if text_input != '':
            if mode in ['t2i', 'd2i', 'e2i', 's2i', 'scr2i', 'cs2i']:
                if mode == 'cs2i':
                    image_cond, image_cond_source = predict_compodiff(None, text_input, negative_prompt, cfg_image_scale, cfg_text_scale, mask=mask_pil, random_seed=random_seed)
                    image_cond_color_compensation, _ = predict_compodiff(image_source, text_input, negative_prompt, cfg_image_scale, cfg_text_scale, mask=mask_pil, random_seed=random_seed)
                    image_cond = 0.9 * image_cond + 0.1 * image_cond_color_compensation
                else:
                    image_cond, image_cond_source = predict_compodiff(None, text_input, negative_prompt, cfg_image_scale, cfg_text_scale, mask=mask_pil, random_seed=random_seed)
            else:
                image_cond, image_cond_source = predict_compodiff(image_source, text_input, negative_prompt, cfg_image_scale, cfg_text_scale, mask=mask_pil, random_seed=random_seed)
            image_cond = image_cond.to(torch.float16).to('cuda:0')
            image_cond_source = image_cond_source.to(torch.float16).to('cuda:0')
        else:
            image_cond = torch.zeros([1, 1, 768]).to(torch.float16).to('cuda:0')

        if image_cond_source is None and mode != 't2i':
            image_cond_source = image_source.resize((512, 512))
            image_cond_source = model_dict['clip_preprocess'](image_cond_source, return_tensors='pt')['pixel_values'].to('cuda:0')
            image_cond_source = model_dict['clip_model'].encode_images(image_cond_source)

        if cfg_image_reference_mix_weight > 0.0 and torch.sum(image_cond_reference).item() != 0.0:
            if torch.sum(image_cond).item() == 0.0:
                image_cond = image_cond_reference
            else:
                image_cond = (1.0 - cfg_image_reference_mix_weight) * image_cond + cfg_image_reference_mix_weight * image_cond_reference

        if cfg_image_source_mix_weight > 0.0:
            image_cond = (1.0 - cfg_image_source_mix_weight) * image_cond + cfg_image_source_mix_weight * image_cond_source

        if negative_prompt != '':
            negative_image_cond, _ = predict_compodiff(None, negative_prompt, '', cfg_image_scale, cfg_text_scale, mask=mask_pil, random_seed=random_seed)
            negative_image_cond = negative_image_cond.to(torch.float16).to('cuda:0')
        else:
            negative_image_cond = torch.zeros_like(image_cond)

        # negative_prompt_embeds
        image_source = torch.tensor(np.array(image_source))
        depth_map = einops.repeat(generate_depth_map(image_source, height, width), 'n c h w -> n (c r) h w', r=3).float().cpu()

        images = model_dict['pipe'](text_input,
                                    image=image_source,
                                    mask=mask,
                                    depth_map=depth_map,
                                    num_inference_steps=int(steps),
                                    image_cond_embeds=image_cond,
                                    negative_image_cond_embeds=negative_image_cond,
                                    guidance_scale=cfg_image_space_scale,
                                    use_depth_map_as_input=use_depth_map_as_input,
                                    apply_mask_to_input=use_mask,
                                    mode=mode,
                                    generator=torch.manual_seed(random_seed),
                                    num_images_per_prompt=2).images
        if do_sr:
            images = model_dict['sr_model'].predict(images)

    return images, [org_image_source, mask_pil, to_pil_image(0.5 * (depth_map[0] + 1.0))]


def generate_canvas(image):
    return Image.fromarray((np.ones([512, 512, 3]) * 255).astype('uint8'))


def surprise_me():
    return random.sample(model_dict['prompt_candidates'], k=1)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Demo')
    parser.add_argument('--model_folder', default=None, type=str, help='path to model_folder')

    args = parser.parse_args()


    global model_dict

    model_dict = build_models(args)

    ### define gradio demo
    title = 'Graphit demo'

    md_title = f'''# {title}
    Diffusion on GPU.
    '''
    neg_default = 'watermark, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
    with gr.Blocks(title=title) as demo:
        gr.Markdown(md_title)
        mode_t2i = gr.Textbox(value='t2i', label='mode selection', visible=False)
        mode_i2i = gr.Textbox(value='i2i', label='mode selection', visible=False)
        mode_inpaint = gr.Textbox(value='inpaint', label='mode selection', visible=False)
        mode_s2i = gr.Textbox(value='s2i', label='mode selection', visible=False)
        mode_scr2i = gr.Textbox(value='scr2i', label='mode selection', visible=False)
        mode_d2i = gr.Textbox(value='d2i', label='mode selection', visible=False)
        mode_e2i = gr.Textbox(value='e2i', label='mode selection', visible=False)
        mode_inped = gr.Textbox(value='inped', label='mode selection', visible=False)
        mode_cs2i = gr.Textbox(value='cs2i', label='mode selection', visible=False)
        mask_scale_default = gr.Number(value=1.0, label='mask scale', visible=False)
        use_edge_default = gr.Checkbox(value=True, label='use color map with edge map', visible=False)
        height_default = gr.Number(value=512, precision=0, label='height', visible=False)
        width_default = gr.Number(value=512, precision=0, label='width', visible=False)
        with gr.Row():
            with gr.Column():
                with gr.Tabs():
                    '''
                    image to image
                    inpainting
                    depth to image
                    saliency map to image
                    '''
                    with gr.TabItem("Text to Image"):
                        image_source_t2i = gr.Image(type='pil', label='Source image', visible=False)
                        with gr.Row():
                            steps_input_t2i = gr.Radio(['5', '10', '25', '50'], value='25', label='denoising steps')
                            random_seed_t2i = gr.Number(value=12345, precision=0, label='Seed')
                        with gr.Accordion('Advanced options', open=False):
                            with gr.Row():
                                cfg_image_scale_t2i = gr.Number(value=1.1, label='attn source image scale', visible=False)
                                cfg_image_space_scale_t2i = gr.Number(value=7.5, label='attn image space scale')
                                cfg_text_scale_t2i = gr.Number(value=7.5, label='attn text scale')
                            negative_text_input_t2i = gr.Textbox(value=neg_default, label='Negative text')
                        with gr.Row():
                            cfg_image_source_mix_weight_t2i = gr.Number(value=0.0, label='weight for mixing source image (0.0~1.0)', visible=False)
                            cfg_image_reference_mix_weight_t2i = gr.Number(value=0.65, label='weight for mixing reference image (0.0~1.0)')
                        with gr.Row():
                            height_t2i = gr.Number(value=512, precision=0, label='height (~512)')
                            width_t2i = gr.Number(value=512, precision=0, label='width (~512)')
                        submit_button_t2i = gr.Button('Generate images')
                    with gr.TabItem("Image to Image"):
                        image_source_i2i = gr.Image(type='pil', label='Source image')
                        with gr.Row():
                            steps_input_i2i = gr.Radio(['5', '10', '25', '50'], value='25', label='denoising steps')
                            random_seed_i2i = gr.Number(value=12345, precision=0, label='Seed')
                        with gr.Accordion('Advanced options', open=False):
                            with gr.Row():
                                cfg_image_scale_i2i = gr.Number(value=1.1, label='attn source image scale', visible=False)
                                cfg_image_space_scale_i2i = gr.Number(value=7.5, label='attn image space scale')
                                cfg_text_scale_i2i = gr.Number(value=7.5, label='attn text scale')
                            negative_text_input_i2i = gr.Textbox(value=neg_default, label='Negative text')
                        with gr.Row():
                            cfg_image_source_mix_weight_i2i = gr.Number(value=0.05, label='weight for mixing source image (0.0~1.0)')
                            cfg_image_reference_mix_weight_i2i = gr.Number(value=0.65, label='weight for mixing reference image (0.0~1.0)')
                        submit_button_i2i = gr.Button('Generate images')
                    with gr.TabItem("Depth to Image"):
                        image_source_d2i = gr.Image(type='pil', label='Source image')
                        with gr.Row():
                            steps_input_d2i = gr.Radio(['5', '10', '25', '50'], value='25', label='denoising steps')
                            random_seed_d2i = gr.Number(value=12345, precision=0, label='Seed')
                        with gr.Accordion('Advanced options', open=False):
                            with gr.Row():
                                cfg_image_scale_d2i = gr.Number(value=1.1, label='attn source image scale', visible=False)
                                cfg_image_space_scale_d2i = gr.Number(value=7.5, label='attn image space scale')
                                cfg_text_scale_d2i = gr.Number(value=7.5, label='attn text scale')
                            negative_text_input_d2i = gr.Textbox(value=neg_default, label='Negative text')
                        with gr.Row():
                            cfg_image_source_mix_weight_d2i = gr.Number(value=0.0, label='weight for mixing source image (0.0~1.0)', visible=False)
                            cfg_image_reference_mix_weight_d2i = gr.Number(value=1.0, label='weight for mixing reference image (0.0~1.0)')
                        submit_button_d2i = gr.Button('Generate images')
                    with gr.TabItem("Edge to Image"):
                        image_source_e2i = gr.Image(type='pil', label='Source image')
                        with gr.Row():
                            steps_input_e2i = gr.Radio(['5', '10', '25', '50'], value='25', label='denoising steps')
                            random_seed_e2i = gr.Number(value=12345, precision=0, label='Seed')
                        with gr.Accordion('Advanced options', open=False):
                            with gr.Row():
                                cfg_image_scale_e2i = gr.Number(value=1.1, label='attn source image scale', visible=False)
                                cfg_image_space_scale_e2i = gr.Number(value=7.5, label='attn image space scale')
                                cfg_text_scale_e2i = gr.Number(value=7.5, label='attn text scale')
                            negative_text_input_e2i = gr.Textbox(value=neg_default, label='Negative text')
                        with gr.Row():
                            cfg_image_source_mix_weight_e2i = gr.Number(value=0.0, label='weight for mixing source image (0.0~1.0)', visible=False)
                            cfg_image_reference_mix_weight_e2i = gr.Number(value=1.0, label='weight for mixing reference image (0.0~1.0)')
                        submit_button_e2i = gr.Button('Generate images')
                    with gr.TabItem("Inpaint"):
                        image_source_inp = gr.Image(type='pil', label='Source image', tool='sketch')
                        with gr.Row():
                            steps_input_inp = gr.Radio(['5', '10', '25', '50'], value='25', label='denoising steps')
                            random_seed_inp = gr.Number(value=12345, precision=0, label='Seed')
                        with gr.Accordion('Advanced options', open=False):
                            with gr.Row():
                                cfg_image_scale_inp = gr.Number(value=1.1, label='attn source image scale', visible=False)
                                cfg_image_space_scale_inp = gr.Number(value=7.5, label='attn image space scale')
                                cfg_text_scale_inp = gr.Number(value=7.5, label='attn text scale')
                            negative_text_input_inp = gr.Textbox(value='', label='Negative text')
                        with gr.Row():
                            cfg_image_source_mix_weight_inp = gr.Number(value=0.0, label='weight for mixing source image (0.0~1.0)', visible=False)
                            cfg_image_reference_mix_weight_inp = gr.Number(value=0.65, label='weight for mixing reference image (0.0~1.0)')
                        submit_button_inp = gr.Button('Generate images')
                    with gr.TabItem("Blending"):
                        image_source_inped = gr.Image(type='pil', label='Source image', tool='sketch')
                        with gr.Row():
                            steps_input_inped = gr.Radio(['5', '10', '25', '50'], value='25', label='denoising steps')
                            random_seed_inped = gr.Number(value=12345, precision=0, label='Seed')
                        with gr.Accordion('Advanced options', open=False):
                            with gr.Row():
                                cfg_image_scale_inped = gr.Number(value=1.1, label='attn source image scale', visible=False)
                                cfg_image_space_scale_inped = gr.Number(value=7.5, label='attn image space scale')
                                cfg_text_scale_inped = gr.Number(value=7.5, label='attn text scale')
                            negative_text_input_inped = gr.Textbox(value=neg_default, label='Negative text')
                        with gr.Row():
                            cfg_image_source_mix_weight_inped = gr.Number(value=0.0, label='weight for mixing source image (0.0~1.0)', visible=False)
                            cfg_image_reference_mix_weight_inped = gr.Number(value=0.35, label='weight for mixing reference image (0.0~1.0)')
                        with gr.Row():
                            mask_scale_inped = gr.Number(value=1.0, label='edge scale')
                            use_edge_inped = gr.Checkbox(value=False, label='use a color map with an edge map')
                        submit_button_inped = gr.Button('Generate images')
                    with gr.TabItem("Sketch (Rough) to Image"):
                        with gr.Column():
                            image_source_s2i = gr.Image(type='pil', label='Source image', tool='sketch', brush_radius=100).style(height=256, width=256)
                            build_canvas_s2i = gr.Button('Build canvas')
                        with gr.Row():
                            steps_input_s2i = gr.Radio(['5', '10', '25', '50'], value='25', label='denoising steps')
                            random_seed_s2i = gr.Number(value=12345, precision=0, label='Seed')
                        with gr.Accordion('Advanced options', open=False):
                            with gr.Row():
                                cfg_image_scale_s2i = gr.Number(value=1.1, label='attn source image scale', visible=False)
                                cfg_image_space_scale_s2i = gr.Number(value=7.5, label='attn image space scale')
                                cfg_text_scale_s2i = gr.Number(value=7.5, label='attn text scale')
                            negative_text_input_s2i = gr.Textbox(value=neg_default, label='Negative text')
                        with gr.Row():
                            cfg_image_source_mix_weight_s2i = gr.Number(value=0.0, label='weight for mixing source image (0.0~1.0)', visible=False)
                            cfg_image_reference_mix_weight_s2i = gr.Number(value=0.65, label='weight for mixing reference image (0.0~1.0)')
                            mask_scale_s2i = gr.Number(value=0.5, label='sketch weight (0.0~1.0)')
                        submit_button_s2i = gr.Button('Generate images')
                    with gr.TabItem("Sketch (Detail) to Image"):
                        with gr.Column():
                            image_source_scr2i = gr.Image(type='pil', label='Source image', tool='sketch', brush_radius=10).style(height=256, width=256)
                            build_canvas_scr2i = gr.Button('Build canvas')
                        with gr.Row():
                            steps_input_scr2i = gr.Radio(['5', '10', '25', '50'], value='25', label='denoising steps')
                            random_seed_scr2i = gr.Number(value=12345, precision=0, label='Seed')
                        with gr.Accordion('Advanced options', open=False):
                            with gr.Row():
                                cfg_image_scale_scr2i = gr.Number(value=1.1, label='attn source image scale', visible=False)
                                cfg_image_space_scale_scr2i = gr.Number(value=7.5, label='attn image space scale')
                                cfg_text_scale_scr2i = gr.Number(value=7.5, label='attn text scale')
                            negative_text_input_scr2i = gr.Textbox(value=neg_default, label='Negative text')
                        with gr.Row():
                            cfg_image_source_mix_weight_scr2i = gr.Number(value=0.0, label='weight for mixing source image (0.0~1.0)', visible=False)
                            cfg_image_reference_mix_weight_scr2i = gr.Number(value=0.65, label='weight for mixing reference image (0.0~1.0)')
                            mask_scale_scr2i = gr.Number(value=0.5, label='sketch weight (0.0~1.0)')
                        submit_button_scr2i = gr.Button('Generate images')
                    with gr.TabItem("Color Sketch to Image"):
                        with gr.Column():
                            image_source_cs2i = gr.Image(type='pil', source='canvas', label='Source image', tool='color-sketch').style(height=256, width=256)
                            #build_canvas_cs2i = gr.Button('Build canvas')
                        with gr.Row():
                            steps_input_cs2i = gr.Radio(['5', '10', '25', '50'], value='25', label='denoising steps')
                            random_seed_cs2i = gr.Number(value=12345, precision=0, label='Seed')
                        with gr.Accordion('Advanced options', open=False):
                            with gr.Row():
                                cfg_image_scale_cs2i = gr.Number(value=1.1, label='attn source image scale', visible=False)
                                cfg_image_space_scale_cs2i = gr.Number(value=7.5, label='attn image space scale')
                                cfg_text_scale_cs2i = gr.Number(value=7.5, label='attn text scale')
                            negative_text_input_cs2i = gr.Textbox(value=neg_default, label='Negative text')
                        with gr.Row():
                            cfg_image_source_mix_weight_cs2i = gr.Number(value=0.0, label='weight for mixing source image (0.0~1.0)', visible=False)
                            cfg_image_reference_mix_weight_cs2i = gr.Number(value=0.65, label='weight for mixing reference image (0.0~1.0)')
                            mask_scale_cs2i = gr.Number(value=0.5, label='sketch weight (0.0~1.0)')
                        submit_button_cs2i = gr.Button('Generate images')
                    text_input = gr.Textbox(value='', label='Input text')
                    submit_surprise_me = gr.Button('Surprise me')
                #swap_button = gr.Button('Swap source with reference', visible=False)
            with gr.Column():
                with gr.Row():
                    do_sr = gr.Checkbox(value=False, label='Super-resolution')
                image_reference = gr.Image(type='pil', label='Reference image')
                gallery_outputs = gr.Gallery(label='Generated outputs').style(grid=[2], height='auto')
                gallery_inputs = gr.Gallery(label='Processed inputs').style(grid=[2], height='auto')

        submit_button_t2i.click(generate, inputs=[image_source_t2i, image_reference, text_input, negative_text_input_t2i, steps_input_t2i, random_seed_t2i, cfg_image_scale_t2i, cfg_text_scale_t2i, cfg_image_space_scale_t2i, cfg_image_reference_mix_weight_t2i, cfg_image_source_mix_weight_t2i, mask_scale_default, use_edge_default, height_t2i, width_t2i, do_sr, mode_t2i], outputs=[gallery_outputs, gallery_inputs])
        submit_button_i2i.click(generate, inputs=[image_source_i2i, image_reference, text_input, negative_text_input_i2i, steps_input_i2i, random_seed_i2i, cfg_image_scale_i2i, cfg_text_scale_i2i, cfg_image_space_scale_i2i, cfg_image_reference_mix_weight_i2i, cfg_image_source_mix_weight_i2i, mask_scale_default, use_edge_default, height_default, width_default, do_sr, mode_i2i], outputs=[gallery_outputs, gallery_inputs])
        submit_button_d2i.click(generate, inputs=[image_source_d2i, image_reference, text_input, negative_text_input_d2i, steps_input_d2i, random_seed_d2i, cfg_image_scale_d2i, cfg_text_scale_d2i, cfg_image_space_scale_d2i, cfg_image_reference_mix_weight_d2i, cfg_image_source_mix_weight_d2i, mask_scale_default, use_edge_default, height_default, width_default, do_sr, mode_d2i], outputs=[gallery_outputs, gallery_inputs])
        submit_button_e2i.click(generate, inputs=[image_source_e2i, image_reference, text_input, negative_text_input_e2i, steps_input_e2i, random_seed_e2i, cfg_image_scale_e2i, cfg_text_scale_e2i, cfg_image_space_scale_e2i, cfg_image_reference_mix_weight_e2i, cfg_image_source_mix_weight_e2i, mask_scale_default, use_edge_default, height_default, width_default, do_sr, mode_e2i], outputs=[gallery_outputs, gallery_inputs])
        submit_button_inp.click(generate, inputs=[image_source_inp, image_reference, text_input, negative_text_input_inp, steps_input_inp, random_seed_inp, cfg_image_scale_inp, cfg_text_scale_inp, cfg_image_space_scale_inp, cfg_image_reference_mix_weight_inp, cfg_image_source_mix_weight_inp, mask_scale_default, use_edge_default, height_default, width_default, do_sr, mode_inpaint], outputs=[gallery_outputs, gallery_inputs])
        submit_button_inped.click(generate, inputs=[image_source_inped, image_reference, text_input, negative_text_input_inped, steps_input_inped, random_seed_inped, cfg_image_scale_inped, cfg_text_scale_inped, cfg_image_space_scale_inped, cfg_image_reference_mix_weight_inped, cfg_image_source_mix_weight_inped, mask_scale_inped, use_edge_inped, height_default, width_default, do_sr, mode_inped], outputs=[gallery_outputs, gallery_inputs])
        submit_button_s2i.click(generate, inputs=[image_source_s2i, image_reference, text_input, negative_text_input_s2i, steps_input_s2i, random_seed_s2i, cfg_image_scale_s2i, cfg_text_scale_s2i, cfg_image_space_scale_s2i, cfg_image_reference_mix_weight_s2i, cfg_image_source_mix_weight_s2i, mask_scale_s2i, use_edge_default, height_default, width_default, do_sr, mode_s2i], outputs=[gallery_outputs, gallery_inputs])
        submit_button_scr2i.click(generate, inputs=[image_source_scr2i, image_reference, text_input, negative_text_input_scr2i, steps_input_scr2i, random_seed_scr2i, cfg_image_scale_scr2i, cfg_text_scale_scr2i, cfg_image_space_scale_scr2i, cfg_image_reference_mix_weight_scr2i, cfg_image_source_mix_weight_scr2i, mask_scale_scr2i, use_edge_default, height_default, width_default, do_sr, mode_scr2i], outputs=[gallery_outputs, gallery_inputs])
        submit_button_cs2i.click(generate, inputs=[image_source_cs2i, image_reference, text_input, negative_text_input_cs2i, steps_input_cs2i, random_seed_cs2i, cfg_image_scale_cs2i, cfg_text_scale_cs2i, cfg_image_space_scale_cs2i, cfg_image_reference_mix_weight_cs2i, cfg_image_source_mix_weight_cs2i, mask_scale_cs2i, use_edge_default, height_default, width_default, do_sr, mode_cs2i], outputs=[gallery_outputs, gallery_inputs])
        build_canvas_s2i.click(generate_canvas, inputs=[image_source_s2i], outputs=[image_source_s2i])
        build_canvas_scr2i.click(generate_canvas, inputs=[image_source_scr2i], outputs=[image_source_scr2i])
        submit_surprise_me.click(surprise_me, outputs=[text_input])
    demo.queue()
    demo.launch(server_name='0.0.0.0',
                server_port=8800)
