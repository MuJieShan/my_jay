#bin/bash
export _HF_DEFAULT_ENDPOINT="https://hf-mirror.com"
export HF_ENDPOINT="https://hf-mirror.com"
model="bert-base-uncased"
target_ratio=0.5
epoch=5
batchsize=64
weight_decay=0.002
learning_rate=3e-5
task=('mnli')
for dataset in "${task[@]}"
do
    echo "运行dataset=$dataset"
    python_command="python ../../scoretrain.py --state ft --dataset $dataset --seed 3404 --reg 5e-8 --weight_decay $weight_decay --epoch0 1 --epoch $epoch --remain_loss 1 --model $model --target_ratio 0 --pruneFlag up --batchsize $batchsize --optim adamw_torch --learning_rate $learning_rate"
    eval $python_command
    python_command="python ../../scoretrain.py --state ft --dataset $dataset --seed 3404 --reg 5e-8 --weight_decay $weight_decay --epoch0 2 --epoch $epoch --remain_loss 1 --model $model --target_ratio 0 --pruneFlag up --batchsize $batchsize --optim adamw_torch --learning_rate $learning_rate"
    eval $python_command
    python_command="python ../../scoretrain.py --state ft --dataset $dataset --seed 3404 --reg 5e-8 --weight_decay $weight_decay --epoch0 2 --epoch $epoch --remain_loss 1 --model $model --target_ratio 0 --pruneFlag up --batchsize $batchsize --optim adamw_torch --learning_rate $learning_rate"
    eval $python_command
done
