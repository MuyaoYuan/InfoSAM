#!/usr/bin/env bash

# Set common paths
DATA_PATH="./data/isic2017"
TRAIN_SEQ_PATH="train.csv"
TEST_SEQ_PATH="test.csv"
MODEL_BASE_PATH="exp_infoSAM/isic"
TRAIN_MODEL_PATH="$MODEL_BASE_PATH/train/InfoSAM_b_isic"
TEST_MODEL_PATH="$MODEL_BASE_PATH/test/InfoSAM_b_isic"

BACKEND="mutable_efficient_vit_b"
SNAPSHOT_PATH="ckpt/sam_vit_b_01ec64.pth"
ADAPTER_CONFIG="-1 -1 0 0 0 0 0 0 0 0 0 0"
MLP_CONFIG="0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25"

python -u train_attention.py \
    --dataset=isic \
    --data-path="$DATA_PATH" \
    --sequence-path="$TRAIN_SEQ_PATH" \
    --split=train \
    --mask_num=1 \
    --models-path="$TRAIN_MODEL_PATH" \
    --batch-size=4 \
    --epochs=10 \
    --start_lr=2e-4 \
    --gt_loss=StructureLoss \
    --tuning_decoder \
    --backend_t vit_b \
    --ckpt_t ckpt/sam_vit_b_01ec64.pth \
    --backend "$BACKEND" \
    --snapshot "$SNAPSHOT_PATH" \
    --adapter_config $ADAPTER_CONFIG \
    --mlp_config $MLP_CONFIG \
    --no_multimask \
    --unfrozen_norm=True \
    --rkd_type=dualmi \
    --relation_type=attn 

for epoch in {6..10}
do
    python -u val.py \
        --dataset=isic \
        --data-path="$DATA_PATH" \
        --sequence-path="$TEST_SEQ_PATH" \
        --split=test \
        --mask_num=1 \
        --save-path="$TEST_MODEL_PATH/epoch_$epoch" \
        --batch-size=1 \
        --backend "$BACKEND" \
        --adapter_config $ADAPTER_CONFIG \
        --mlp_config $MLP_CONFIG \
        --no_multimask \
        --snapshot "$TRAIN_MODEL_PATH/${BACKEND}_${epoch}_.pth" 
done
