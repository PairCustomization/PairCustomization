from diffusers import StableDiffusionXLPipeline, AutoencoderKL
import torch
import os
import sys
sys.path.append(".")
from modules import *

def run_evaluation():
    lora_path_1_name = "./example_loras/lora-dog-digital-art-style.safetensors"
    lora_path_2_name = "./example_loras/lora-man-painting-style.safetensors"
    lora_epoch_1 = 750
    lora_epoch_2 = 750
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        subfolder=None,
        # cache_dir="your-path-to-vae-cache-dir",
    )

    SDXL_pipeline = StableDiffusionXLPipelineLoraGuidance.from_pretrained ("stabilityai/stable-diffusion-xl-base-1.0", vae=vae) 
    SDXL_pipeline.to("cuda")

    SDXL_pipeline.load_lora_weights(lora_path_1_name, adapter_name="style_weights_digital_art")
    SDXL_pipeline.load_lora_weights(lora_path_2_name, adapter_name="style_weights_painting")

    prompts = [["A photo of a dog", ["in digital art style", "in painting style"]], ["A photo of a cat", ["in digital art style", "in painting style"]]]
    os.makedirs("test_outputs", exist_ok=True)
    random_seed = 1234123
    for prompt, lora_prompt_add_on in prompts:
        lora_prompt_art, lora_prompt_painting = lora_prompt_add_on
        # run inference with both styles
        torch.manual_seed(random_seed)
        images = SDXL_pipeline(prompt=prompt,
                            lora_prompt_add_on=[lora_prompt_art, lora_prompt_painting], 
                            num_inference_steps=50,
                            guidance_scale=7.5,
                            lora_guidance_scale=[4.5, 4.5],
                            start_LoRA_step=800,
                            lora_name=["style_weights_digital_art", "style_weights_painting"]
                            ).images[0]
        images.save(f"test_outputs/test_output_{prompt.replace(' ', '_')}_seed_{random_seed}_both_styles.png")
        # reset adapters in model
        SDXL_pipeline.set_adapters("style_weights_digital_art", 0)
        SDXL_pipeline.set_adapters("style_weights_painting", 0)
        # run inference with digital art style
        torch.manual_seed(random_seed)
        images = SDXL_pipeline(prompt=prompt,
                            lora_prompt_add_on=lora_prompt_art, 
                            num_inference_steps=50,
                            guidance_scale=7.5,
                            lora_guidance_scale=5.5,
                            lora_name="style_weights_digital_art",
                            ).images[0]
        images.save(f"test_outputs/test_output_{prompt.replace(' ', '_')}_seed_{random_seed}_digital_art_style.png")
        SDXL_pipeline.set_adapters("style_weights_digital_art", 0)
        SDXL_pipeline.set_adapters("style_weights_painting", 0)
        # run inference with painting style
        torch.manual_seed(random_seed)
        images = SDXL_pipeline(prompt=prompt,
                            lora_prompt_add_on=lora_prompt_painting,
                            num_inference_steps=50,
                            guidance_scale=7.5,
                            lora_guidance_scale=5.5,
                            lora_name="style_weights_painting"
                            ).images[0]
        images.save(f"test_outputs/test_output_{prompt.replace(' ', '_')}_seed_{random_seed}_painting_style.png")
        # run inference with no style
        torch.manual_seed(random_seed)
        images = SDXL_pipeline(prompt=prompt,
                            lora_prompt_add_on=None,
                            num_inference_steps=50,
                            guidance_scale=7.5,
                            lora_guidance_scale=None,
                            ).images[0]
        images.save(f"test_outputs/test_output_{prompt.replace(' ', '_')}_seed_{random_seed}_no_lora.png")
if __name__ == "__main__":
    run_evaluation()