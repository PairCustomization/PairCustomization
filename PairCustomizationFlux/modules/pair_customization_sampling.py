from modules.src.flux.modules.layers import (
    SingleStreamBlockProcessor,
    DoubleStreamBlockProcessor,
    SingleStreamBlockLoraProcessor,
    DoubleStreamBlockLoraProcessor,
)

from modules.src.flux.sampling import get_noise, get_schedule, prepare, unpack, prepare_img, prepare_prompt

from modules.src.flux.util import (
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    load_controlnet,
    load_flow_model_quintized,
    Annotator,
    get_lora_rank,
    load_checkpoint
)

import torch
from torch import Tensor
from PIL import Image
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from einops import rearrange, repeat

class PipelinePairCustomization:
    def __init__(self, ae, model, device, clip=None, t5=None, prospective_prompts=None, offload=False):
        self.clip = clip
        self.t5 = t5
        self.ae = ae
        self.model = model
        self.model.eval()
        self.device = device
        self.prospective_prompts = prospective_prompts
        self.offload = offload

        if offload and self.prospective_prompts is not None:
            self.prospective_prompts = {prompt: prepare_prompt(prompt, t5, clip) for prompt in self.prospective_prompts}
            del self.clip
            del self.t5
            self.clip = None
            self.t5 = None
            torch.cuda.empty_cache()

    def change_weights_lora_scale(self, dit, lora_weight_scale):
        for name, module in dit.named_modules():
            if isinstance(module, DoubleStreamBlockLoraProcessor) or isinstance(module, SingleStreamBlockLoraProcessor):
                # print(f"changing weights scale of {name} to {lora_weight_scale}")
                module.lora_weight = lora_weight_scale

    def set_lora(self, local_path: str = None, repo_id: str = None,
                 name: str = None, lora_weight: int = 1.0):
        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.update_model_with_lora(checkpoint, lora_weight)

    def set_lora_from_collection(self, lora_type: str = "realism", lora_weight: int = 1.0):
        checkpoint = load_checkpoint(
            None, self.hf_lora_collection, self.lora_types_to_names[lora_type]
        )
        self.update_model_with_lora(checkpoint, lora_weight)

    def update_model_with_lora(self, checkpoint, lora_weight):
        rank = get_lora_rank(checkpoint)
        lora_attn_procs = {}

        for name, _ in self.model.attn_processors.items():
            lora_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    lora_state_dict[k[len(name) + 1:]] = checkpoint[k]

            if len(lora_state_dict):
                if name.startswith("single_blocks"):
                    lora_attn_procs[name] = SingleStreamBlockLoraProcessor(dim=3072, rank=rank, lora_weight=lora_weight)
                else:
                    lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(dim=3072, rank=rank, lora_weight=lora_weight)
                lora_attn_procs[name].load_state_dict(lora_state_dict)
                lora_attn_procs[name].to(self.device)
            else:
                if name.startswith("single_blocks"):
                    lora_attn_procs[name] = SingleStreamBlockProcessor()
                else:
                    lora_attn_procs[name] = DoubleStreamBlockProcessor()

        self.model.set_attn_processor(lora_attn_procs)

    def denoise_pair_customization(
        self,
        model,
        # model input
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        vec: Tensor,
        txt_style,
        txt_ids_style,
        vec_style,
        timesteps: list[float],
        neg_txt: Tensor = None,
        neg_txt_ids: Tensor = None,
        neg_vec: Tensor = None,
        guidance: float = 4.0,
        true_gs = 1,
        timestep_to_start_cfg=float("inf"),
        timestep_to_start_style_guidance=6,
        style_guidance_scale=4.0,
    ):
        i = 0
        # this is ignored for schnell

        guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            # print(f"setting weights scale to 0")
            self.change_weights_lora_scale(model, 0.0)
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            pred = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
            )

            if i >= timestep_to_start_cfg:
                neg_pred = model(
                    img=img,
                    img_ids=img_ids,
                    txt=neg_txt,
                    txt_ids=neg_txt_ids,
                    y=neg_vec,
                    timesteps=t_vec,
                    guidance=guidance_vec, 
                )
            else:
                neg_pred = pred 
                
            if i >= timestep_to_start_style_guidance:
                # print(f"changing weights scale")
                self.change_weights_lora_scale(model, 1.0)
                style_pred = model(
                    img=img,
                    img_ids=img_ids,
                    txt=txt_style,
                    txt_ids=txt_ids_style,
                    y=vec_style,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )
                self.change_weights_lora_scale(model, 0.0)
            else:
                style_pred = pred
            pred = neg_pred + true_gs * (pred - neg_pred) + style_guidance_scale * (style_pred - pred)

            img = img + (t_prev - t_curr) * pred
            i += 1
        return img

    def __call__(self,
                 prompt: str,
                 style_prompt: str,
                 image_prompt: Image = None,
                 width: int = 512,
                 height: int = 512,
                 guidance: float = 4,
                 num_steps: int = 50,
                 seed: int = 123456789,
                 true_gs: float = 1.0,
                 neg_prompt = None,
                 timestep_to_start_cfg = float("inf"),
                 timestep_to_start_style_guidance=6,
                 style_guidance_scale=4.0,
                 ):
        width = 16 * (width // 16)
        height = 16 * (height // 16)

        return self.forward(
            prompt,
            style_prompt,
            width,
            height,
            guidance,
            num_steps,
            seed,
            timestep_to_start_cfg=timestep_to_start_cfg,
            true_gs=true_gs,
            neg_prompt=neg_prompt,
            timestep_to_start_style_guidance=timestep_to_start_style_guidance,
            style_guidance_scale=style_guidance_scale,
        )

    @torch.inference_mode()
    def gradio_generate(self, prompt, image_prompt, controlnet_image, width, height, guidance,
                        num_steps, seed, true_gs, ip_scale, neg_ip_scale, neg_prompt,
                        neg_image_prompt, timestep_to_start_cfg, control_type, control_weight,
                        lora_weight, local_path, lora_local_path, ip_local_path):
        if controlnet_image is not None:
            controlnet_image = Image.fromarray(controlnet_image)
            if ((self.controlnet_loaded and control_type != self.control_type)
                or not self.controlnet_loaded):
                if local_path is not None:
                    self.set_controlnet(control_type, local_path=local_path)
                else:
                    self.set_controlnet(control_type, local_path=None,
                                        repo_id=f"xlabs-ai/flux-controlnet-{control_type}-v3",
                                        name=f"flux-{control_type}-controlnet-v3.safetensors")
        if lora_local_path is not None:
            self.set_lora(local_path=lora_local_path, lora_weight=lora_weight)
        if image_prompt is not None:
            image_prompt = Image.fromarray(image_prompt)
            if neg_image_prompt is not None:
                neg_image_prompt = Image.fromarray(neg_image_prompt)
            if not self.ip_loaded:
                if ip_local_path is not None:
                    self.set_ip(local_path=ip_local_path)
                else:
                    self.set_ip(repo_id="xlabs-ai/flux-ip-adapter",
                                name="flux-ip-adapter.safetensors")
        seed = int(seed)
        if seed == -1:
            seed = torch.Generator(device="cpu").seed()

        img = self(prompt, image_prompt, controlnet_image, width, height, guidance,
                   num_steps, seed, true_gs, control_weight, ip_scale, neg_ip_scale, neg_prompt,
                   neg_image_prompt, timestep_to_start_cfg)

        filename = f"output/gradio/{uuid.uuid4()}.jpg"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Make] = "XLabs AI"
        exif_data[ExifTags.Base.Model] = self.model_type
        img.save(filename, format="jpeg", exif=exif_data, quality=95, subsampling=0)
        return img, filename

    def forward(
        self,
        prompt,
        style_prompt,
        width,
        height,
        guidance,
        num_steps,
        seed,
        controlnet_image = None,
        timestep_to_start_cfg = 0,
        true_gs = 3.5,
        control_weight = 0.9,
        neg_prompt="",
        timestep_to_start_style_guidance=6,
        style_guidance_scale=4.0,
    ):
        x = get_noise(
            1, height, width, device=self.device,
            dtype=torch.bfloat16, seed=seed
        )
        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )
        torch.manual_seed(seed)
        with torch.no_grad():
            if self.offload:
                if self.t5 is not None:
                    self.t5.to(self.device)
                if self.clip is not None:
                    self.clip.to(self.device)
            if self.t5 is not None and self.clip is not None:            
                inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)
                if timestep_to_start_cfg < len(timesteps):
                    neg_inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt)
                else:
                    neg_inp_cond = {'txt': None, 'txt_ids': None}
                style_inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=style_prompt)
            else:
                assert prompt in self.prospective_prompts, f"Prompt {prompt} not found in prospective_prompts"
                assert style_prompt in self.prospective_prompts, f"Prompt {style_prompt} not found in prospective_prompts"
                if neg_prompt is not None:
                    assert neg_prompt in self.prospective_prompts, f"Prompt {neg_prompt} not found in prospective_prompts"
                
                inp_cond = self.prospective_prompts[prompt]
                style_inp_cond = self.prospective_prompts[style_prompt]
                if neg_prompt is not None:
                    neg_inp_cond = self.prospective_prompts[neg_prompt]
                else:
                    neg_inp_cond = {'txt': None, 'txt_ids': None, 'vec': None}
                    
                img_inputs = prepare_img(x)

            if self.offload:
                if self.t5 is not None:
                    self.offload_model_to_cpu(self.t5)
                if self.clip is not None:
                    self.offload_model_to_cpu(self.clip)
                self.model = self.model.to(self.device)
            
                x = self.denoise_pair_customization(
                    self.model,
                    img=img_inputs['img'],
                    img_ids=img_inputs['img_ids'],
                    txt=inp_cond['txt'],
                    txt_ids=inp_cond['txt_ids'],
                    vec=inp_cond['vec'],
                    txt_style=style_inp_cond['txt'],
                    txt_ids_style=style_inp_cond['txt_ids'],
                    vec_style=style_inp_cond['vec'],
                    timesteps=timesteps,
                    guidance=guidance,
                    timestep_to_start_cfg=timestep_to_start_cfg,
                    neg_txt=neg_inp_cond['txt'],
                    neg_txt_ids=neg_inp_cond['txt_ids'],
                    neg_vec=neg_inp_cond['vec'],
                    true_gs=true_gs,
                    timestep_to_start_style_guidance=timestep_to_start_style_guidance,
                    style_guidance_scale=style_guidance_scale,
                )

            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(x.device)
            x = unpack(x.float(), height, width)
            x = self.ae.decode(x)
            self.offload_model_to_cpu(self.ae.decoder)

            # self.model.to(x.device)

        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
        return output_img

    def offload_model_to_cpu(self, *models):
        if not self.offload: return
        for model in models:
            model.cpu()
            torch.cuda.empty_cache()

class SamplerPairCustomization(PipelinePairCustomization):
    def __init__(self, ae, model, device, clip=None, t5=None, prospective_prompts=None, offload=False):
        self.clip = clip
        self.t5 = t5
        self.ae = ae
        self.model = model
        self.model.eval()
        self.device = device
        self.prospective_prompts = prospective_prompts
        self.offload = offload

        if offload and self.prospective_prompts is not None:
            self.prospective_prompts = {prompt: prepare_prompt(t5, clip, prompt) for prompt in self.prospective_prompts}
            del self.clip
            del self.t5
            torch.cuda.empty_cache()
