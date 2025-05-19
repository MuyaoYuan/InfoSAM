#!/usr/bin/env bash

PY_ARGS=${@:1}

# Set common paths
DATA_PATH="./data/leaf_disease_segmentation"
TRAIN_SEQ_PATH="train.csv"
TEST_SEQ_PATH="test.csv"
MODEL_BASE_PATH="exp_infoSAM/leaf"
TEST_MODEL_PATH="$MODEL_BASE_PATH/test/InfoSAM_b_leaf"

BACKEND="mutable_efficient_vit_b"
ADAPTER_CONFIG="-1 -1 0 0 0 0 0 0 0 0 0 0"
MLP_CONFIG="0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25"

python -u val.py \
    --dataset=leaf \
    --data-path="$DATA_PATH" \
    --sequence-path="$TEST_SEQ_PATH" \
    --split=test \
    --mask_num=1 \
    --save-path="$TEST_MODEL_PATH" \
    --batch-size=1 \
    --backend "$BACKEND" \
    --adapter_config $ADAPTER_CONFIG \
    --mlp_config $MLP_CONFIG \
    --no_multimask \
    ${PY_ARGS}
