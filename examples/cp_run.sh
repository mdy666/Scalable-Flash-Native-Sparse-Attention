export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS=8

# ["ag", "swa", "slc", "cmp", "topk", "nsa", "1M"]
RUN_FUNC="ag"

TOTAL_SEQLEN=$((1024 * 128))
MEAN_SEQLEN=$((1024 * 128))
STD_SEQLEN=0

NUM_HEADS=64
NUM_KV_HEADS=4
# qk_head_dim can set 192
QK_HEAD_DIM=128
V_HEAD_DIM=128

# For varlen use cp_mode=1 or 2, for non-varlen use cp_mode=2
CP_MODE=2

rm -rf ./examples/print_data/*

torchrun --master-port=29501 --nproc-per-node=$GPUS test_cp.py \
    --func $RUN_FUNC \
    --seqlen $TOTAL_SEQLEN \
    --mean $MEAN_SEQLEN \
    --std $STD_SEQLEN \
    --num-heads $NUM_HEADS \
    --num-kv-heads $NUM_KV_HEADS \
    --qk-head-dim $QK_HEAD_DIM \
    --v-head-dim $V_HEAD_DIM \
    --cp-mode $CP_MODE $@

rm -rf ./examples/print_data/*