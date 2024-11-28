import math
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor
import os

import argparse

import sys
sys.path.append(".")

from modules.src.flux.sampling import denoise, get_noise, get_schedule, prepare, unpack, prepare_img, prepare_prompt
from modules.src.flux.util import (configs, load_ae, load_clip, load_flow_model2, load_t5)
from modules.src.flux.modules.layers import DoubleStreamBlockLoraProcessor, SingleStreamBlockLoraProcessor, DoubleStreamPairedBlockLoraProcessor, SingleStreamPairedBlockLoraProcessor
from modules.src.flux.xflux_pipeline_pair_customization import XFluxSamplerPairCustomization

from modules.pair_customization_sampling import PipelinePairCustomization

def get_models(name: str, device, offload: bool, is_schnell: bool):
    cache_dir = "your-desired-cache-dir" # this variable can be None if you want to save in the base cache directory
    t5 = load_t5(device, max_length=256 if is_schnell else 512, cache_dir=cache_dir)
    clip = load_clip(device, cache_dir=cache_dir)
    clip.requires_grad_(False)
    model = load_flow_model2(name, device="cpu", cache_dir=cache_dir)
    vae = load_ae(name, device="cpu" if offload else device, cache_dir=cache_dir)
    return model, vae, t5, clip

def run_inference(save_folder: str, style: str, device: int = 0):
    neg_prompt = "blurry background" #  bokeh effect
    output_dir = save_folder

    local_path = "example_loras/flux-lora-dog-digital-art-style.safetensors" # this variable is set to where your LoRA model is saved
    checkpoint = 1250
    # if you trained a model, this may be your local path:
    # local_path = f"example_loras/trained-dog-digital-art-style-disjoint/checkpoint-1250/style_lora.safetensors"
    device = torch.device(f"cuda:{device}")
    for content_prompt in ["a photo of a dog", "a photo of a cat", "a professional headshot of a man"]:  
        for seed in range(2):
            prospective_prompts = [content_prompt, f"{content_prompt} {style}", neg_prompt]

            dit, vae, t5, clip = get_models(name="flux-dev", device=device, offload=False, is_schnell=False)

            sampler = PipelinePairCustomization(clip=clip, t5=t5, ae=vae, model=dit, device=device, offload=True, prospective_prompts=prospective_prompts)

            sampler.set_lora(local_path=local_path)

            image = sampler(
                prospective_prompts[0],
                prospective_prompts[1],
                width=512,
                height=512,
                guidance=1, # 2
                num_steps=50,
                seed=seed,
                true_gs=4.0,
                neg_prompt=neg_prompt,
                timestep_to_start_cfg=0,
                timestep_to_start_style_guidance=float("inf"), # 6,
                style_guidance_scale=6.0,
            )
            prompt_object = content_prompt.split(" ")[-1]
            os.makedirs(output_dir, exist_ok=True)
            image.save(f"{output_dir}/{prompt_object}_seed_{seed}_cfg.png")

            image = sampler(
                prospective_prompts[0],
                prospective_prompts[1],
                width=512,
                height=512,
                guidance=1, # 2
                num_steps=50,
                seed=seed,
                true_gs=4.0,
                neg_prompt=neg_prompt,
                timestep_to_start_cfg=0,
                timestep_to_start_style_guidance=10, # 10
                style_guidance_scale=6.0,
            )
            os.makedirs(output_dir, exist_ok=True)
            image.save(f"{output_dir}/{prompt_object}_seed_{seed}_cfg_lora_step_{checkpoint}_style.png")

def main():
    run_inference("lora_output", "digital illustration style", device=0)

if __name__ == '__main__':
    main()