import argparse
import logging
import math
import os
import re
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from safetensors.torch import save_file

import accelerate
# import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from omegaconf import OmegaConf
from copy import deepcopy
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange

import sys
sys.path.append(".")

from modules.src.flux.sampling import denoise, get_noise, get_schedule, prepare, unpack, prepare_img, prepare_prompt
from modules.src.flux.util import (configs, load_ae, load_clip,
                       load_flow_model2, load_t5)
from modules.src.flux.modules.layers import DoubleStreamBlockLoraProcessor, SingleStreamBlockLoraProcessor, DoubleStreamPairedBlockLoraProcessor, SingleStreamPairedBlockLoraProcessor
from modules.src.flux.xflux_pipeline_pair_customization import XFluxSamplerPairCustomization
from modules import *
from modules.image_datasets.dataset import loader
if is_wandb_available():
    import wandb
else:
    wandb = None
logger = get_logger(__name__, log_level="INFO")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_models(name: str, device, offload: bool, is_schnell: bool):
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    clip.requires_grad_(False)
    model = load_flow_model2(name, device="cpu")
    vae = load_ae(name, device="cpu" if offload else device)
    return model, vae, t5, clip

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()

    return args.config

def precompute_orthonormal_bases_flux(seed):
    flux_sizes = [3072, 15360]
    # precompute orthonormal bases for FLUX
    basis_dict = {}
    with torch.no_grad():
        for size in flux_sizes:
            torch.manual_seed(seed)
            basis = torch.randn(size, size)
            basis, _ = torch.linalg.qr(basis)
            basis_dict[size] = basis
    return basis_dict

def set_paired_lora_weights_dit(dit, mode):
    for name, module in dit.named_modules():
        if isinstance(module, DoubleStreamPairedBlockLoraProcessor) or isinstance(module, SingleStreamPairedBlockLoraProcessor):
            module.change_mode(mode)

def reinitialize_lora_style_weights(dit, rank, double_blocks_idx, single_blocks_idx):
    for name, attn_processor in dit.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))
        if name.startswith("double_blocks"):
            if layer_index in double_blocks_idx:
                torch.nn.init.normal_(attn_processor.qkv_lora1.down.weight, std=1 / rank)
                torch.nn.init.normal_(attn_processor.qkv_lora2.down.weight, std=1 / rank)
                torch.nn.init.normal_(attn_processor.proj_lora1.down.weight, std=1 / rank)
                torch.nn.init.normal_(attn_processor.proj_lora2.down.weight, std=1 / rank)
        elif name.startswith("single_blocks"):
            if layer_index in single_blocks_idx:
                torch.nn.init.normal_(attn_processor.qkv_lora.down.weight, std=1 / rank)
                torch.nn.init.normal_(attn_processor.proj_lora.down.weight, std=1 / rank)

def change_weights_lora_scale(dit, double_blocks_idx, single_blocks_idx, lora_weight_scale):
    for name, attn_processor in dit.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))
        if name.startswith("double_blocks"):
            if layer_index in double_blocks_idx:
                attn_processor.lora_weight = lora_weight_scale
        elif name.startswith("single_blocks"):
            if layer_index in single_blocks_idx:
                attn_processor.lora_weight = lora_weight_scale

def change_weights_requires_grad(dit, requires_grad, pattern_match_names):
    if type(pattern_match_names) == str:
        pattern_match_names = [pattern_match_names]
    for name, parameter in dit.named_parameters():
        if all([pattern in name for pattern in pattern_match_names]):
            parameter.requires_grad_(requires_grad)

def add_orthogonal_weights_to_dit(dit, orthonormal_basis_matrices, content_rank, style_rank):
    for name, parameter in dit.named_parameters():            
        if "content" in name:
            if "down" in name:
                # update down weights with orthogonal basis, turn off gradient
                in_features = parameter.shape[1]
                correct_basis = orthonormal_basis_matrices[in_features]
                basis_part = correct_basis[0:content_rank]
                parameter.data = basis_part.contiguous()
                parameter.requires_grad_(False)
        if "style" in name:
            if "down" in name:
                # update down weights with orthogonal basis, turn off gradient
                in_features = parameter.shape[1]
                correct_basis = orthonormal_basis_matrices[in_features]
                basis_part = correct_basis[content_rank:content_rank + style_rank]
                parameter.data = basis_part.contiguous()
                parameter.requires_grad_(False)

def save_content_style_state_dicts(
    unwrapped_model_state_dict, 
    single_blocks_only_content, 
    double_blocks_only_content, 
    single_blocks_only_style, 
    double_blocks_only_style, 
    single_blocks_both,
    double_blocks_both,
    save_path):

    content_lora_state_dict = {}
    for content_only_block_idx in single_blocks_only_content:
        for k, v in unwrapped_model_state_dict.items():
            if f"single_blocks.{content_only_block_idx}." in k and "_lora" in k:
                content_lora_state_dict[k] = v
    for content_only_block_idx in double_blocks_only_content:
        for k, v in unwrapped_model_state_dict.items():
            if f"double_blocks.{content_only_block_idx}." in k and "_lora" in k:
                content_lora_state_dict[k] = v
    for content_only_block_idx in single_blocks_both:
        for k, v in unwrapped_model_state_dict.items():
            if f"single_blocks.{content_only_block_idx}." in k and "_lora_content" in k:
                content_lora_state_dict[k.replace("_lora_content", "_lora")] = v
    for content_only_block_idx in double_blocks_both:
        for k, v in unwrapped_model_state_dict.items():
            if f"double_blocks.{content_only_block_idx}." in k and ("_lora1_content" in k or "_lora2_content" in k):
                content_lora_state_dict[k.replace("_lora1_content", "_lora1").replace("_lora2_content", "_lora2")] = v

    style_lora_state_dict = {}
    for style_only_block_idx in single_blocks_only_style:
        for k, v in unwrapped_model_state_dict.items():
            if f"single_blocks.{style_only_block_idx}." in k and "_lora" in k:
                style_lora_state_dict[k] = v
    for style_only_block_idx in double_blocks_only_style:
        for k, v in unwrapped_model_state_dict.items():
            if f"double_blocks.{style_only_block_idx}." in k and "_lora" in k:
                style_lora_state_dict[k] = v
    for style_only_block_idx in single_blocks_both:
        for k, v in unwrapped_model_state_dict.items():
            if f"single_blocks.{style_only_block_idx}." in k and "_lora_style" in k:
                style_lora_state_dict[k.replace("_lora_style", "_lora")] = v
    for style_only_block_idx in double_blocks_both:
        for k, v in unwrapped_model_state_dict.items():
            if f"double_blocks.{style_only_block_idx}." in k and ("_lora1_style" in k or "_lora2_style" in k):
                style_lora_state_dict[k.replace("_lora1_style", "_lora1").replace("_lora2_style", "_lora2")] = v
    
    os.makedirs(save_path, exist_ok=True)
    save_file(content_lora_state_dict, os.path.join(save_path, "content_lora.safetensors"))
    save_file(style_lora_state_dict, os.path.join(save_path, "style_lora.safetensors"))

def main():

    args = OmegaConf.load(parse_args())
    is_schnell = args.model_name == "flux-schnell"
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        # datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    dit, vae, t5, clip = get_models(name=args.model_name, device=accelerator.device, offload=False, is_schnell=is_schnell)

    content_txt_inps = {prompt: prepare_prompt(t5=t5, clip=clip, prompt=prompt) for prompt in args.data_config.content_img_captions}
    style_txt_inps = {prompt: prepare_prompt(t5=t5, clip=clip, prompt=prompt) for prompt in args.data_config.style_img_captions}



    # print("precomputed prompt tensors, no need for text models")
    del t5
    del clip
    torch.cuda.empty_cache()
    # print("moving sample prompts to cpu until validation loop")

    lora_attn_procs = {}

    orthonormal_basis_matrices = precompute_orthonormal_bases_flux(seed=42)

    if args.double_blocks_content is None:
        double_blocks_content_idx = list(range(19))
    else:
        if args.double_blocks_content == "even":
            double_blocks_content_idx = list(range(0, 19, 2))
        elif args.double_blocks_content == "odd":
            double_blocks_content_idx = list(range(1, 19, 2))
        else:
            double_blocks_content_idx = [int(idx) for idx in args.double_blocks_content.split(",")]

    if args.double_blocks_style is None:
        double_blocks_style_idx = list(range(19))
    else:
        if args.double_blocks_style == "even":
            double_blocks_style_idx = list(range(0, 19, 2))
        elif args.double_blocks_style == "odd":
            double_blocks_style_idx = list(range(1, 19, 2))
        else:
            double_blocks_style_idx = [int(idx) for idx in args.double_blocks_style.split(",")]

    if args.single_blocks_content is None:
        single_blocks_content_idx = list(range(38))
    else:
        if args.single_blocks_content == "even":
            single_blocks_content_idx = list(range(0, 38, 2))
        elif args.single_blocks_content == "odd":
            single_blocks_content_idx = list(range(1, 38, 2))
        else:
            single_blocks_content_idx = [int(idx) for idx in args.single_blocks_content.split(",")]
    
    if args.single_blocks_style is None:
        single_blocks_style_idx = list(range(38))
    else:
        if args.single_blocks_style == "even":
            single_blocks_style_idx = list(range(0, 38, 2))
        elif args.single_blocks_style == "odd":
            single_blocks_style_idx = list(range(1, 38, 2))
        else:
            single_blocks_style_idx = [int(idx) for idx in args.single_blocks_style.split(",")]

    for name, attn_processor in dit.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))

        if name.startswith("double_blocks"):
            # print("setting LoRA Processor for", name)
            content_block, style_block = layer_index in double_blocks_content_idx, layer_index in double_blocks_style_idx
            if content_block and style_block:
                lora_attn_procs[name] = DoubleStreamPairedBlockLoraProcessor(
                dim=3072, rank=args.rank
                )
            elif content_block or style_block:
                lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(
                dim=3072, rank=args.rank
                )
            else:
                lora_attn_procs[name] = attn_processor
        elif name.startswith("single_blocks"):
            # print("setting LoRA Processor for", name)
            content_block, style_block = layer_index in single_blocks_content_idx, layer_index in single_blocks_style_idx
            if content_block and style_block:
                lora_attn_procs[name] = SingleStreamPairedBlockLoraProcessor(
                dim=3072, rank=args.rank
                )
            elif content_block or style_block:
                lora_attn_procs[name] = SingleStreamBlockLoraProcessor(
                dim=3072, rank=args.rank
                )
            else:
                lora_attn_procs[name] = attn_processor
        else:
            lora_attn_procs[name] = attn_processor
    dit.set_attn_processor(lora_attn_procs)

    add_orthogonal_weights_to_dit(dit, orthonormal_basis_matrices, args.rank, args.rank)

    single_blocks_only_style = [i for i in range(38) if i in single_blocks_style_idx and i not in single_blocks_content_idx]
    single_blocks_only_content = [i for i in range(38) if i in single_blocks_content_idx and i not in single_blocks_style_idx]
    single_blocks_both = [i for i in range(38) if i in single_blocks_style_idx and i in single_blocks_content_idx]

    double_blocks_only_style = [i for i in range(19) if i in double_blocks_style_idx and i not in double_blocks_content_idx]
    double_blocks_only_content = [i for i in range(19) if i in double_blocks_content_idx and i not in double_blocks_style_idx]
    double_blocks_both = [i for i in range(19) if i in double_blocks_style_idx and i in double_blocks_content_idx]



    vae.requires_grad_(False)
    # t5.requires_grad_(False)
    # clip.requires_grad_(False)
    dit = dit.to(torch.float32)
    dit.train()
    optimizer_cls = torch.optim.AdamW
    for n, param in dit.named_parameters():
        if '_lora' not in n:
            param.requires_grad = False
        elif 'down' in n and ('content' in n or 'style' in n):
            print(f"Setting {n} to have no down weight grads since it is a paired block")
            param.requires_grad = False
        else:
            print(n)

    print(f"set {len([p.numel() for p in dit.parameters() if p.requires_grad])} parameters to require grad")
    print(sum([p.numel() for p in dit.parameters() if p.requires_grad]) / 1000000, 'parameters')

    grad_parameters = [p for n, p in dit.named_parameters() if p.requires_grad]

    optimizer = optimizer_cls(
        grad_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader = loader(**args.data_config)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    global_step = 0
    first_epoch = 0

    dit, optimizer, _, lr_scheduler = accelerator.prepare(
        dit, optimizer, deepcopy(train_dataloader), lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision


    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    timesteps = get_schedule(
                999,
                (1024 // 8) * (1024 // 8) // 4,
                shift=True,
            )
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_gradient_updates = math.ceil(args.max_train_steps * 1.0 / args.gradient_accumulation_steps * 1.0)
    if wandb is not None:
        wandb.init(
            entity="cmu-gil",
            name="Pair Customization Flux intiial experiment",
            group="Pair Customization Flux",
            project='Pair-Customization-Flux',
        )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f" Number of gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f" Number of gradient updates = {num_gradient_updates}")
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
    progress_bar = tqdm(
        range(0, num_gradient_updates),
        initial=initial_global_step,
        desc="Gradient Updates",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss_content, train_loss_style = 0.0, 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(dit):
                content_img, style_img, content_prompt, style_prompt = batch

                with torch.no_grad():
                    # vae = vae.to(accelerator.device)
                    x_1_content = vae.encode(content_img.to(accelerator.device).to(torch.float32))
                    x_1_style = vae.encode(style_img.to(accelerator.device).to(torch.float32))
                    # vae = vae.to("cpu")
                    # inp = prepare(t5=t5, clip=clip, img=x_1, prompt=prompts)
                    img_inp_content = prepare_img(img=x_1_content)
                    img_inp_style = prepare_img(img=x_1_style)
                    img_inp_content.update(content_txt_inps[content_prompt[0]]) # batch size is 1 for our training
                    img_inp_style.update(style_txt_inps[style_prompt[0]]) # batch size is 1 for our training

                    inp_content = img_inp_content
                    inp_style = img_inp_style

                    x_1_content = rearrange(x_1_content, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
                    x_1_style = rearrange(x_1_style, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

                bs = content_img.shape[0]
                t = torch.tensor([timesteps[random.randint(0, 999)]]).to(accelerator.device)
                noise = torch.randn_like(x_1_content).to(accelerator.device)

                x_t_content = (1 - t) * x_1_content + t * noise

                bsz = x_1_content.shape[0]
                guidance_vec = torch.full((x_t_content.shape[0],), 1, device=x_t_content.device, dtype=x_t_content.dtype)

                set_paired_lora_weights_dit(dit, "content") # set paired blocks to only use content\

                # set content weights in paired blocks to require gradient so they are trained
                change_weights_requires_grad(dit, requires_grad=True, pattern_match_names=["content", "up"]) 
                for content_only_block_idx in single_blocks_only_content:
                    change_weights_requires_grad(dit, requires_grad=True, pattern_match_names=[f"single_blocks.{content_only_block_idx}.", "_lora"]) # set content weights in single blocks to not require gradient so they are not trained
                for content_only_block_idx in double_blocks_only_content:
                    change_weights_requires_grad(dit, requires_grad=True, pattern_match_names=[f"double_blocks.{content_only_block_idx}.", "_lora"]) # set content weights in double blocks to not require gradient so they are not trained
                
                change_weights_lora_scale(dit, double_blocks_only_style, single_blocks_only_style, 0.0) # set style weight blocks to 0.0 so they don't affect generation

                # Predict the noise residual and compute loss
                model_pred_content = dit(img=x_t_content.to(weight_dtype),
                                img_ids=inp_content['img_ids'].to(weight_dtype),
                                txt=inp_content['txt'].to(weight_dtype),
                                txt_ids=inp_content['txt_ids'].to(weight_dtype),
                                y=inp_content['vec'].to(weight_dtype),
                                timesteps=t.to(weight_dtype),
                                guidance=guidance_vec.to(weight_dtype),)

                loss = F.mse_loss(model_pred_content.float(), (noise - x_1_content).float(), reduction="mean")
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss_content += avg_loss.item() / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(dit.parameters(), args.max_grad_norm)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if global_step == args.style_weight_start_training_step:
                    with accelerator.main_process_first():
                        if accelerator.is_main_process:
                            # reinitialize style weights since they may have decayed during the initial content training
                            # there is only one optimizer since deepspeed + accelerate doesn't support multiple optimizers
                            reinitialize_lora_style_weights(dit, args.rank, double_blocks_only_style, single_blocks_only_style)

                if global_step >= args.style_weight_start_training_step:
                    x_t_style = (1 - t) * x_1_style + t * noise
                    set_paired_lora_weights_dit(dit, "both") # set paired blocks to use both content and style
                    
                    # set content weights in paired blocks to not require gradient so they are not trained
                    change_weights_requires_grad(dit, requires_grad=False, pattern_match_names=["content", "up"])
                    for content_only_block_idx in single_blocks_only_content:
                        change_weights_requires_grad(dit, requires_grad=False, pattern_match_names=[f"single_blocks.{content_only_block_idx}.", "_lora"]) # set content weights in single blocks to not require gradient so they are not trained
                    for content_only_block_idx in double_blocks_only_content:
                        change_weights_requires_grad(dit, requires_grad=False, pattern_match_names=[f"double_blocks.{content_only_block_idx}.", "_lora"]) # set content weights in double blocks to not require gradient so they are not trained
                    change_weights_lora_scale(dit, double_blocks_only_style, single_blocks_only_style, 1.0) # turn on style weights for style generation

                    # Predict the noise residual and compute loss
                    model_pred_style = dit(img=x_t_style.to(weight_dtype),
                                    img_ids=inp_style['img_ids'].to(weight_dtype),
                                    txt=inp_style['txt'].to(weight_dtype),
                                    txt_ids=inp_style['txt_ids'].to(weight_dtype),
                                    y=inp_style['vec'].to(weight_dtype),
                                    timesteps=t.to(weight_dtype),
                                    guidance=guidance_vec.to(weight_dtype),)

                    loss = F.mse_loss(model_pred_style.float(), (noise - x_1_style).float(), reduction="mean")
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss_style += avg_loss.item() / args.gradient_accumulation_steps
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(dit.parameters(), args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                accelerator.log({"train_loss_style": train_loss_style}, step=global_step)
                wandb.log({"train_loss_style": train_loss_style}, step=global_step)

                accelerator.log({"train_loss_content": train_loss_content}, step=global_step)
                wandb.log({"train_loss_content": train_loss_content}, step=global_step)

                train_loss_content, train_loss_style = 0.0, 0.0

                if not args.disable_sampling and global_step % args.sample_every == 0:
                    if accelerator.is_main_process:
                        print(f"Sampling images for step {global_step}...")
                        content_text_inputs_copy = {key: value.copy() for key, value in content_txt_inps.items()}
                        style_text_inputs_copy = {key: value.copy() for key, value in style_txt_inps.items()}
                        sampler = XFluxSamplerPairCustomization(clip=None, t5=None, ae=vae, model=dit, device=accelerator.device, offload=False, content_prompt_inps=content_text_inputs_copy, style_prompt_inps=style_text_inputs_copy)
                        set_paired_lora_weights_dit(dit, "content")
                        change_weights_lora_scale(dit, double_blocks_only_content, double_blocks_only_content, 1.0) # set style weight blocks to 0.0 so they don't affect generation
                        change_weights_lora_scale(dit, double_blocks_only_style, single_blocks_only_style, 0.0) # set style weight blocks to 0.0 so they don't affect generation

                        content_images = []
                        for i, prompt in enumerate(args.data_config.content_img_captions):
                            result = sampler(prompt=prompt,
                                             width=args.sample_width,
                                             height=args.sample_height,
                                             num_steps=args.sample_steps,
                                             true_gs=0,
                                             timestep_to_start_cfg=float('inf'),
                                             )
                            content_images.append(wandb.Image(result))
                            os.makedirs("PairCustomization-flux-samples", exist_ok=True)
                            result.save(f"PairCustomization-flux-samples/{global_step}_prompt_{i}_res.png")
                        wandb.log({f"Content results, step {global_step}": content_images})

                        if global_step >= args.style_weight_start_training_step:
                            style_images = []
                            set_paired_lora_weights_dit(dit, "both")
                            change_weights_lora_scale(dit, double_blocks_only_content, double_blocks_only_content, 1.0) # set style weight blocks to 0.0 so they don't affect generation
                            change_weights_lora_scale(dit, double_blocks_only_style, single_blocks_only_style, 1.0) # set style weight blocks to 0.0 so they don't affect generation
                            for i, prompt in enumerate(args.data_config.style_img_captions):
                                result = sampler(prompt=prompt,
                                                width=args.sample_width,
                                                height=args.sample_height,
                                                num_steps=args.sample_steps,
                                                true_gs=0,
                                                timestep_to_start_cfg=float('inf'),
                                                )
                                style_images.append(wandb.Image(result))
                                # print(f"Result for prompt #{i} is generated")
                            os.makedirs("PairCustomization-flux-samples", exist_ok=True)
                            result.save(f"PairCustomization-flux-samples/{global_step}_prompt_{i}_res.png")
                            wandb.log({f"Style results, step {global_step}": style_images})

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                        # accelerator.save_state(save_path)
                        unwrapped_model_state = accelerator.unwrap_model(dit).state_dict()

                        save_content_style_state_dicts(
                            unwrapped_model_state, 
                            single_blocks_only_content, 
                            double_blocks_only_content, 
                            single_blocks_only_style, 
                            double_blocks_only_style, 
                            single_blocks_both,
                            double_blocks_both,
                            save_path)

                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "content_lr": lr_scheduler.get_last_lr()[0], "style_lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()
    wandb.finish()


if __name__ == "__main__":
    main()