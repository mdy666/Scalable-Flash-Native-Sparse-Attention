# ["construct", "mean", "cmp",  "topk", "specific-topk" "slc", "combine", "nsa"]
RUN_FUNC="nsa"

TOTAL_SEQLEN=$((1024 * 8))
MEAN_SEQLEN=$((1024 * 4))
STD_SEQLEN=1024

NUM_HEADS=64
NUM_KV_HEADS=4
# qk_head_dim can set 192
QK_HEAD_DIM=128
V_HEAD_DIM=128

python test.py \
    --func $RUN_FUNC \
    --seqlen $TOTAL_SEQLEN \
    --mean $MEAN_SEQLEN \
    --std $STD_SEQLEN \
    --num-heads $NUM_HEADS \
    --num-kv-heads $NUM_KV_HEADS \
    --qk-head-dim $QK_HEAD_DIM \
    --v-head-dim $V_HEAD_DIM $@
