export LORA_NAME="full_weights_rank_32"
export SAVE_FOLDER="output_$LORA_NAME"
export CHECKPOINT=1250
export STYLE="digital illustration style"
export DEVICE="0"

python inference/run_inference.py --device $DEVICE --lora_name $LORA_NAME --save_folder $SAVE_FOLDER --checkpoint $CHECKPOINT --style "$STYLE"