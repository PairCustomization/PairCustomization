export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_CONTENT_DIR="data/dog_digital_art_style/real/"
export INSTANCE_STYLE_DIR="data/dog_digital_art_style/styled/"
export OUTPUT_DIR="example_loras/lora-dog-digital-art-style"
export INSTANCE_PROMPT_SUBJECT="A photo of a sbu dog"
export INSTANCE_PROMPT_STYLE="in digital art style"
export INSTANCE_VALIDATION_PROMPT="default" 
# when using "default" for INSTANCE_VALIDATION_PROMPT, 
# validation images will be generated both using INSTANCE_PROMPT_SUBJECT using the content weights
# and f"{INSTANCE_PROMPT_SUBJECT} {INSTANCE_PROMPT_STYLE}" using the content and style weights (for training step > style_weight_start_training_step)
# if INSTANCE_VALIDATION_PROMPT is not default, both weights will be used for the prompt
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
# to specify a specific cache directory for the VAE model, add the following argument: 
#   --pretrained_vae_model_cache_dir="your-path-to-vae-cache-dir" \


accelerate launch training_scripts/train_pair_customization_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_content_data_dir=$INSTANCE_CONTENT_DIR \
  --instance_styled_data_dir=$INSTANCE_STYLE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt_subject="$INSTANCE_PROMPT_SUBJECT" \
  --instance_prompt_style="$INSTANCE_PROMPT_STYLE" \
  --resolution=512 \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-5 \
  --content_rank=64 \
  --style_rank=64 \
  --report_to="tensorboard" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=750 \
  --checkpointing_steps=750 \
  --style_weight_start_training_step=250 \
  --validation_prompt="$INSTANCE_VALIDATION_PROMPT" \
  --validation_epochs=250 \
  --seed="0" \
  --fix_down_weight_orthogonal