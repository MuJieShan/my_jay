#bin/bash
export _HF_DEFAULT_ENDPOINT="https://hf-mirror.com"
export HF_ENDPOINT="https://hf-mirror.com"
dataset='all'
model="bert-base-uncased"
target_ratio=0.5

epoch=5
batchsize=64
weight_decay=0.002
learning_rate=3e-5
task=('mnli')
for t in "${task[@]}"
do
    echo "运行task=$t"
    python_command="python ../../traindata3_0.py --state ft --dataset $t --seed 3404 --reg 5e-8 --weight_decay $weight_decay --epoch0 1 --epoch $epoch --remain_loss 1 --model $model --target_ratio $target_ratio --pruneFlag up --batchsize $batchsize --learning_rate $learning_rate"
    eval $python_command
    python_command="python ../../traindata3_0.py --state ft --dataset $t --seed 3404 --reg 5e-8 --weight_decay $weight_decay --epoch0 1 --epoch $epoch --remain_loss 1 --model $model --target_ratio $target_ratio --pruneFlag down --batchsize $batchsize --learning_rate $learning_rate"
    eval $python_command
done