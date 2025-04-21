#!/bin/bash
GPU=$1
DATASET=$2
MODEL=$3
MAX_PARALLEL=2

BATCH_SIZE=32
SLIDE_WIN=5
dim=64
out_layer_num=1
SLIDE_STRIDE=1
topk=5
out_layer_inter_dim=128
val_ratio=0.2
decay=1e-4
lr=0.001
early_stop_win=10
loss_func="contrastive"

COMMENT="${DATASET}"

if [ "$DATASET" == "wadi" ]; then
    dim=128
    topk=30
    out_layer_inter_dim=128
    SLIDE_WIN=10
elif [ "$DATASET" == "swat" ]; then
    dim=64
    topk=15
    out_layer_inter_dim=64
fi

OUT_DIR="./results/${DATASET}_${MODEL}_seeds"
mkdir -p $OUT_DIR

EPOCH=50
report='best'

pids=()
for seed in $(seq 1 10); do
    (
    echo "Running seed $seed..."

    CUDA_VISIBLE_DEVICES=$GPU python main.py \
        -dataset $DATASET \
        -save_path_pattern "${DATASET}_${MODEL}_seed${seed}" \
        -slide_stride $SLIDE_STRIDE \
        -slide_win $SLIDE_WIN \
        -batch $BATCH_SIZE \
        -epoch $EPOCH \
        -comment $COMMENT \
        -random_seed $seed \
        -loss_func $loss_func \
        -lr $lr \
        -early_stop_win $early_stop_win \
        -decay $decay \
        -dim $dim \
        -out_layer_num $out_layer_num \
        -out_layer_inter_dim $out_layer_inter_dim \
        -val_ratio $val_ratio \
        -report $report \
        -topk $topk \
        -model_type $MODEL

    RESULT_DIR="./results/${DATASET}_${MODEL}_seed${seed}"
    LATEST_CSV=$(ls -t $RESULT_DIR/*.csv 2>/dev/null | head -n 1)

    if [ $seed -eq 1 ]; then
        echo "seed,F1,Precision,Recall,Accuracy,AUC" > $OUT_DIR/summary.csv
    fi

    if [ -f "$LATEST_CSV" ]; then
        result_line=$(tail -n 1 "$LATEST_CSV")
        echo "$seed,$result_line" >> $OUT_DIR/summary.csv
    else
        echo "$seed,N/A,N/A,N/A,N/A,N/A" >> $OUT_DIR/summary.csv
    fi
    ) &

    pids+=($!)

    if [[ $((${#pids[@]} % $MAX_PARALLEL)) -eq 0 ]]; then
        wait "${pids[@]}"
        pids=()
    fi
done

wait "${pids[@]}"
