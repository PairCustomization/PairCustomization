#!usr/bin/bash

# export CONFIG_FILE="training/test_pair_customization_disjiont.yaml"
export CONFIG_FILE="training/test_pair_customization_original.yaml"

accelerate launch --config_file training/accelerate_config.yaml training/train_pair_customization_lora_flux.py --config $CONFIG_FILE