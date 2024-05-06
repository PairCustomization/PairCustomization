from diffusers import StableDiffusionXLPipeline, AutoencoderKL, ControlNetModel
import torch
import os
import sys
from diffusers.utils import load_image
sys.path.append(".")
from modules import *
import cv2
from PIL import Image
import numpy as np

def run_evaluation():
    lora_path_name = "./example_loras/lora-dog-digital-art-style.safetensors"
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        # cache_dir="your-path-to-vae-cache-dir",
        subfolder=None,
    )

    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0") # cache_dir="your-path-to-controlnet-cache-dir",
    SDXL_pipeline = StableDiffusionXLControlNetPipelineLoraGuidance.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae, controlnet=controlnet) 
    SDXL_pipeline.to("cuda")
    SDXL_pipeline.load_lora_weights(lora_path_name, adapter_name="style_weights")

    prompts = [("A professional headshot of a man", "in digital art style")]
    image_files = ["./data/man_painting_style/real/image.png"] # image that we will use to generate the edgemap
    os.makedirs("test_outputs", exist_ok=True)
    random_seed = 1234123
    for (prompt, lora_prompt_add_on), image_file in zip(prompts, image_files):
        image = np.array(load_image(image_file).resize((1024, 1024)))
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None].repeat(3, axis=-1)
        image = Image.fromarray(image)
        image.save(f"test_outputs/canny_image_test_output_{prompt.replace(' ', '_')}.png")
        SDXL_pipeline.set_adapters(["style_weights"], 1)
        torch.manual_seed(random_seed)
        guidance_scale = 5.0
        lora_guidance_scale = 6.0
        images = SDXL_pipeline(prompt=prompt, 
                            lora_prompt_add_on=lora_prompt_add_on, 
                            num_inference_steps=50,
                            controlnet_conditioning_scale=0.5,
                            image=image,
                            guidance_scale=guidance_scale,
                            lora_guidance_scale=lora_guidance_scale,
                            start_LoRA_step=1000,
                            lora_name="style_weights"
                            ).images[0]
        images.save(f"test_outputs/controlnet_test_output_{prompt.replace(' ', '_')}_guidance_{guidance_scale}_lora_{lora_guidance_scale}_seed_{random_seed}.png")

if __name__ == "__main__":
    run_evaluation()