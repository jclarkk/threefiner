import math
import numpy as np
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoencoderKL,
    DDIMScheduler
)
from diffusers.utils.import_utils import is_xformers_available
from transformers import logging

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd


class StableDiffusionXL(nn.Module):
    def __init__(self,
                 device,
                 sd_version,
                 fp16=True,
                 vram_O=False,
                 hf_key=None,
                 use_refiner=False,
                 sd_mode=None,
                 ssd_M=100,
                 t_range=[0.02, 0.50]):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading SDXL...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
            model_key_refiner = hf_key
        elif 'turbo' in self.sd_version:
            model_key = "stabilityai/sdxl-turbo"
            use_refiner = False
        elif self.sd_version == 'xl-1.0':
            model_key = "stabilityai/stable-diffusion-xl-base-1.0"
            model_key_refiner = "stabilityai/stable-diffusion-xl-refiner-1.0"
        else:
            model_key = "stabilityai/stable-diffusion-xl-base-0.9"
            model_key_refiner = "stabilityai/stable-diffusion-xl-refiner-0.9"

        self.precision_t = torch.float16 if fp16 else torch.float32
        variant = "fp16" if fp16 else None

        # Create model
        cache_dir = "pretrained/SDXL"
        general_kwargs = {
            "cache_dir": cache_dir,
            "torch_dtype": self.precision_t,
            "use_safetensors": True,
            "variant": variant,
            # "local_files_only": True,
        }
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            # "pretrained/SDXL/vae_fp16",
            # local_files_only=True,
            use_safetensors=True,
            torch_dtype=self.precision_t
        )
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_key,
            vae=vae,
            **general_kwargs
        )

        if use_refiner:
            refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_key_refiner,
                                                                       **general_kwargs)
            self.refiner = refiner.to("cuda")

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            if is_xformers_available():
                pipe.enable_xformers_memory_efficient_attention()
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2
        self.unet = pipe.unet
        # self.scheduler = pipe.scheduler
        Scheduler = DDIMScheduler

        self.scheduler = Scheduler.from_pretrained(
            model_key,
            cache_dir=cache_dir,
            subfolder="scheduler",
            torch_dtype=self.precision_t,
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])

        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.embeddings = {}

        # Score distillation mode
        self.sd_mode = sd_mode
        self.ssd_M = ssd_M

        print(f'[INFO] loaded SDXL!')

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        prompt_embeds_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                prompt_embeds = text_encoder(text_inputs.input_ids.to(self.device),
                                             output_hidden_states=True, )
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
                prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # Do the same for unconditional embeddings
        negative_prompt_embeds_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            uncond_input = tokenizer(
                negative_prompts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                negative_prompt_embeds = text_encoder(uncond_input.input_ids.to(self.device),
                                                      output_hidden_states=True, )
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
                negative_prompt_embeds_list.append(negative_prompt_embeds)

        negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        self.embeddings['pos'] = prompt_embeds
        self.embeddings['pooled_pos'] = pooled_prompt_embeds
        self.embeddings['neg'] = negative_prompt_embeds
        self.embeddings['pooled_neg'] = negative_pooled_prompt_embeds

        # directional embeddings
        for d in ['front', 'side', 'back']:
            embeds = self.encode_text([f'{p}, {d} view' for p in prompts])
            self.embeddings[d] = embeds

    def encode_text(self, prompt):
        # prompt: [str]
        inputs = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length,
                                truncation=True, return_tensors="pt")
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    def train_step(
            self,
            pred_rgb,
            step_ratio=None,
            guidance_scale=100,
            latents=None,
            grad_scale=1.0,
            weight_choice=0,
            vers=None,
            hors=None,
    ):

        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.precision_t)

        B = pred_rgb.size(0)

        if latents is None:
            pred_rgb_1024 = F.interpolate(pred_rgb, (1024, 1024), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_1024)
        else:
            latents = F.interpolate(latents, (128, 128), mode='bilinear', align_corners=False)

        negative_prompt_embeds = self.embeddings['neg']#.expand(batch_size, -1, -1)
        negative_pooled_prompt_embeds = self.embeddings['pooled_neg']#.expand(batch_size, -1, -1)
        prompt_embeds = self.embeddings['pos']#.expand(batch_size, -1, -1)
        pooled_prompt_embeds = self.embeddings['pooled_pos']#.expand(batch_size, -1, -1)

        add_text_embeds = pooled_prompt_embeds
        res = 1024  # if self.opt.latent else self.opt.res_fine
        add_time_ids = self._get_add_time_ids(
            (res, res), (0, 0), (res, res), dtype=prompt_embeds.dtype
        ).repeat_interleave(B, dim=0)


        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0).to(self.device)

        with torch.no_grad():

            if step_ratio is not None:
                # dreamtime-like
                # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
                t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
                t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            else:
                t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            t_ = torch.cat([t] * 2, dim=0)
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            noise_pred = self.unet(
                latent_model_input,
                t_,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t)
        if weight_choice == 0:
            w = (1 - self.alphas[t])[:, None, None, None]  # sigma_t^2
        elif weight_choice == 1:
            w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
            w = w[:, None, None, None]
        elif weight_choice == 2:  # check official code in Fantasia3D
            w = 1 / (1 - self.alphas[t])
            w = w[:, None, None, None]
        else:
            w = 1

        if self.sd_mode == 'csd':
            grad = w * (noise_pred_text - noise_pred_uncond) * grad_scale
        elif self.sd_mode == 'ssd':
            h = w * (noise_pred_text - noise_pred_uncond)
            r = (noise_pred_text * noise).sum() / noise_norm(noise)
            noise_tilde = noise_pred_text - r * noise
            E_ssd = noise_norm(h) / noise_norm(noise_tilde) * noise_tilde
            grad = h
            grad[t <= self.ssd_M] = E_ssd[t <= self.ssd_M]
            grad = grad * grad_scale
        else:
            grad = w * (noise_pred - noise) * grad_scale

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)

        return loss

    def set_timesteps(self, num_inference_steps, last_t):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, optional):
                the device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        # Clipping the minimum of all lambda(t) for numerical stability.
        # This is critical for cosine (squaredcos_cap_v2) noise schedule.
        clipped_idx = torch.searchsorted(torch.flip(self.scheduler.lambda_t, [0]),
                                         self.scheduler.config.lambda_min_clipped)
        last_timestep = ((last_t - clipped_idx).cpu().numpy()).item()

        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        if self.scheduler.config.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, last_timestep - 1, num_inference_steps + 1).round()[::-1][:-1].copy().astype(np.int64)
            )
        elif self.scheduler.config.timestep_spacing == "leading":
            step_ratio = last_timestep // (num_inference_steps + 1)
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps + 1) * step_ratio).round()[::-1][:-1].copy().astype(np.int64)
            timesteps += self.scheduler.config.steps_offset
        elif self.scheduler.config.timestep_spacing == "trailing":
            step_ratio = self.scheduler.config.num_train_timesteps / num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.arange(last_timestep, 0, -step_ratio).round().copy().astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.scheduler.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )

        sigmas = np.array(((1 - self.scheduler.alphas_cumprod) / self.scheduler.alphas_cumprod) ** 0.5)
        self.scheduler.sigmas = torch.from_numpy(sigmas).to(self.device)

        # when num_inference_steps == num_train_timesteps, we can end up with
        # duplicates in timesteps.
        _, unique_indices = np.unique(timesteps, return_index=True)
        timesteps = timesteps[np.sort(unique_indices)]

        self.scheduler.timesteps = torch.from_numpy(timesteps).to(self.device)

        self.scheduler.num_inference_steps = len(timesteps)

        self.scheduler.model_outputs = [
                                           None,
                                       ] * self.scheduler.config.solver_order
        self.scheduler.lower_order_nums = 0

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def decode_latents(self, latents, refiner_vae=False):

        if refiner_vae:
            vae = self.refiner.vae.to(torch.float32)
        else:
            vae = self.vae

        latents = 1 / vae.config.scaling_factor * latents

        with torch.no_grad():
            imgs = vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def w_star(t, m1=800, m2=500, s1=300, s2=100):
    # max time 1000
    r = np.ones_like(t) * 1.0
    r[t > m1] = np.exp(-((t[t > m1] - m1) ** 2) / (2 * s1 * s1))
    r[t < m2] = np.exp(-((t[t < m2] - m2) ** 2) / (2 * s2 * s2))
    return r


def precompute_prior(T=1000, min_t=200, max_t=800):
    ts = np.arange(T)
    prior = w_star(ts)[min_t:max_t]
    prior = prior / prior.sum()
    prior = prior[::-1].cumsum()[::-1]
    return prior, min_t


def time_prioritize(step_ratio, time_prior, min_t=200):
    return np.abs(time_prior - step_ratio).argmin() + min_t


def noise_norm(eps):
    # [B, 3, H, W]
    return torch.sqrt(torch.square(eps).sum(dim=[1, 2, 3]))
