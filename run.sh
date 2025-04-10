gpu_n=$1
DATASET=$2
MODEL=$3

seed=42
BATCH_SIZE=32
SLIDE_WIN=5
dim=64
out_layer_num=1
SLIDE_STRIDE=1
topk=5
out_layer_inter_dim=128
val_ratio=0.2
decay=0


path_pattern="${DATASET}"
COMMENT="${DATASET}"

if [ "$DATASET" == "wadi" ]; then
    dim=128
    topk=30
    out_layer_inter_dim=128
elif [ "$DATASET" == "swat" ]; then
    dim=64
    topk=15
    out_layer_inter_dim=64
fi

EPOCH=50
report='val'

if [[ "$gpu_n" == "cpu" ]]; then
    python main.py \
        -dataset $DATASET \
        -save_path_pattern $path_pattern \
        -slide_stride $SLIDE_STRIDE \
        -slide_win $SLIDE_WIN \
        -batch $BATCH_SIZE \
        -epoch $EPOCH \
        -comment $COMMENT \
        -random_seed $seed \
        -decay $decay \
        -dim $dim \
        -out_layer_num $out_layer_num \
        -out_layer_inter_dim $out_layer_inter_dim \
        -decay $decay \
        -val_ratio $val_ratio \
        -report $report \
        -topk $topk \
        -model_type $MODEL \
        -device 'cpu'
else
    CUDA_VISIBLE_DEVICES=$gpu_n  python main.py \
        -dataset $DATASET \
        -save_path_pattern $path_pattern \
        -slide_stride $SLIDE_STRIDE \
        -slide_win $SLIDE_WIN \
        -batch $BATCH_SIZE \
        -epoch $EPOCH \
        -comment $COMMENT \
        -random_seed $seed \
        -decay $decay \
        -dim $dim \
        -out_layer_num $out_layer_num \
        -out_layer_inter_dim $out_layer_inter_dim \
        -decay $decay \
        -val_ratio $val_ratio \
        -report $report \
        -topk $topk \
        -model_type $MODEL \
fi