#!/bin/bash
# ====== RRSIS_SAM3 Training Script ======
# Optimized for Kaggle T4/P100 (16GB VRAM)
#
# Usage:
#   bash fine.sh <dataset_name>
#   e.g., bash fine.sh rrsis_d
#         bash fine.sh rrsis_hr
#         bash fine.sh refsegrs

DATASET=${1:-rrsis_d}
DATA_ROOT=${2:-./data}
OUTPUT_DIR="./output/${DATASET}_sam3"

echo "============================================="
echo "  RRSIS_SAM3 Training"
echo "  Dataset: ${DATASET}"
echo "  Data Root: ${DATA_ROOT}"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================="

python train.py \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --sam3_ckpt ./pre-trained-weights/sam3.pt \
    --image_size 504 \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --epochs 40 \
    --batch_size 2 \
    --grad_accum_steps 4 \
    --lr 5e-5 \
    --lr_backbone 1e-5 \
    --lr_decoder 5e-5 \
    --weight_decay 0.01 \
    --warmup_epochs 5 \
    --fp16 \
    --gradient_checkpointing \
    --seed 42 \
    --num_workers 4
