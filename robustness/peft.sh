export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="peft-sd-pokemon-model"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

accelerate launch  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --checkpointing_steps=500 \
  --validation_prompt="A pokemon with blue eyes." \
  --seed=1337