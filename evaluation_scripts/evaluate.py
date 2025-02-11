from diffusers import StableDiffusionXLPipeline, AutoencoderKL
import torch
import os
import sys
sys.path.append(".")
from modules import *

def run_evaluation():
    lora_path_name = "./example_loras/lora-dog-digital-art-style.safetensors"
    lora_epoch = 750
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        # cache_dir="your-path-to-vae-cache-dir",
        subfolder=None,
    )

    SDXL_pipeline = StableDiffusionXLPipelineLoraGuidance.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae) # 
    SDXL_pipeline.to("cuda")
    SDXL_pipeline.load_lora_weights(lora_path_name, adapter_name="style_weights")

    prompts = [("A photo of a dog", "in digital art style"), ("A photo of a cat", "in digital art style")]
    os.makedirs("test_outputs", exist_ok=True)
    random_seed = 1234123
    for prompt, lora_prompt_add_on in prompts:
        SDXL_pipeline.set_adapters(["style_weights"], 1)
        torch.manual_seed(random_seed)
        images = SDXL_pipeline(prompt=prompt, 
                            lora_prompt_add_on=lora_prompt_add_on, 
                            num_inference_steps=50,
                            guidance_scale=5.0,
                            lora_guidance_scale=4.0,
                            start_LoRA_step=800,
                            lora_name="style_weights"
                            ).images[0]
        images.save(f"test_outputs/test_output_{prompt.replace(' ', '_')}_seed_{random_seed}.png")
        SDXL_pipeline.set_adapters("style_weights", 0)
        torch.manual_seed(random_seed)
        images = SDXL_pipeline(prompt=prompt, 
                            lora_prompt_add_on=None, 
                            num_inference_steps=50,
                            guidance_scale=5.0,
                            lora_guidance_scale=None,
                            lora_name=None
                            ).images[0]
        images.save(f"test_outputs/test_output_{prompt.replace(' ', '_')}_seed_{random_seed}_no_lora.png")
if __name__ == "__main__":
    run_evaluation()
