import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
import sys
sys.path.append(".")
from modules import *
from PIL import Image

# For more complex images, I recommend a lower start_LoRA step and higher lora_guidance_scale. 
# For simpler images (i.e. one object in the foreground only), I recommend a higher start_LoRA step and lower lora_guidance_scale.

def get_man_digital_art_parameters():
    return {
        "prompt": "A professional headshot of a man",
        "style_add_on": "in digital art style",
        "lora_path_name": "./example_loras/lora-dog-digital-art-style.safetensors",
        "image_path": "./data/man_painting_style/real/image.png",
        "inversion_max_step": 1,
        "edit_cfg": 5.0,
        "guidance_scale": 5.0,
        "lora_guidance_scale": 4.0,
        "start_LoRA_step": 1000,
        "num_inference_steps": 50,
        "seed": 7865,
    }

def get_dog_painting_style_parameters():
    return {
        "prompt": "A photo of a dog outside",
        "style_add_on": "in painting style",
        "lora_path_name": "./example_loras/lora-man-painting-style.safetensors",
        "image_path": "./data/dog_digital_art_style/real/dog.png",
        "inversion_max_step": 1,
        "edit_cfg": 5.0,
        "guidance_scale": 5.0,
        "lora_guidance_scale": 6.0,
        "start_LoRA_step": 600,
        "num_inference_steps": 50,
        "seed": 7865,
    }

def run_evaluation():
    # either add the dog digital art style to the man image, or add the man painting style to the dog image:
     
    parameters = get_man_digital_art_parameters()
    # parameters = get_dog_painting_style_parameters()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_type = "SDXL"
    scheduler_type = "DDIM"
    pipe_inversion, pipe_inference = ReNoiseGetPipes(model_type, scheduler_type, device=device)
    del pipe_inference
    os.makedirs("test_outputs", exists_ok=True)

    config = ReNoiseRunConfig(model_type = model_type,
                        num_inference_steps = parameters["num_inference_steps"],
                        num_inversion_steps = parameters["num_inference_steps"],
                        num_renoise_steps = 1,
                        scheduler_type = scheduler_type,
                        perform_noise_correction = False,
                        inversion_max_step = parameters["inversion_max_step"],
                        seed = parameters["seed"])

    _, inv_latent, _, all_latents = ReNoiseinvert(Image.open(parameters["image_path"]),
                                            parameters["prompt"],
                                            config,
                                            pipe_inversion=pipe_inversion,
                                            pipe_inference=None,
                                            do_reconstruction=False,
                                            edit_cfg=parameters["edit_cfg"],
                                            )
    del pipe_inversion
    torch.cuda.empty_cache()

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        cache_dir="your-path-to-vae-cache-dir",
        subfolder=None,
    )
    pipe_inference = StableDiffusionXLImg2ImgPipelineLoraGuidance.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae)
    pipe_inference.scheduler = DDIMScheduler.from_config(pipe_inference.scheduler.config)
    pipe_inference.to(torch.device("cuda")).to(dtype=torch.float16)

    pipe_inference.load_lora_weights(parameters["lora_path_name"], adapter_name="style_weights")
    pipe_inference.set_adapters(["style_weights"], 1)

    edit_img = pipe_inference(
        prompt=parameters["prompt"],
        negative_prompt=parameters["prompt"],
        lora_prompt_add_on=parameters["style_add_on"],
        num_inference_steps=parameters["num_inference_steps"],
        strength=1.0,
        guidance_scale=parameters["edit_cfg"],
        denoising_start=0.0,
        lora_guidance_scale=parameters["lora_guidance_scale"],
        start_LoRA_step=parameters["start_LoRA_step"],
        lora_name="style_weights",
        image=inv_latent,
    ).images[0]

    prompt, guidance_scale, lora_guidance_scale, start_step = parameters["prompt"], parameters["guidance_scale"], parameters["lora_guidance_scale"], parameters["start_LoRA_step"]

    edit_img.save(f"test_outputs/real_image_test_output_{prompt.replace(' ', '_')}_guidance_{guidance_scale}_lora_{lora_guidance_scale}_start_LoRA_step_{start_step}.png")

if __name__ == "__main__":
    run_evaluation()
