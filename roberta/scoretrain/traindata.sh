#bin/bash
export _HF_DEFAULT_ENDPOINT="https://hf-mirror.com"
export HF_ENDPOINT="https://hf-mirror.com"
model="bert-base-uncased"
target_ratio=0.5
epoch=10
batchsize=32
weight_decay=0

task=("mrpc" "cola" "rte" "stsb" "sst2")
for dataset in "${task[@]}"
do
    echo "运行dataset=$dataset"
    python_command="python ../../scoretrain.py --state ft --dataset $dataset --seed 3404 --reg 5e-8 --weight_decay $weight_decay --epoch0 5 --epoch $epoch --remain_loss 1 --model $model --target_ratio 0 --pruneFlag up --batchsize $batchsize --optim adamw_torch"
    eval $python_command
done

