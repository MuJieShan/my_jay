#bin/bash
export _HF_DEFAULT_ENDPOINT="https://hf-mirror.com"
export HF_ENDPOINT="https://hf-mirror.com"

dataset='all'
model="roberta-base"
target_ratio=0.5

epoch=10
batchsize=32
weight_decay=0
learning_rate=2e-5
task=('sst2')
for t in "${task[@]}"
do
    echo "运行task=$t"
    python_command="python ../../traindata12_0.py --state ft --dataset $t --seed 3404 --reg 5e-7 --weight_decay $weight_decay --epoch0 1 --epoch $epoch --remain_loss 1 --model $model --target_ratio $target_ratio --pruneFlag up --batchsize $batchsize --learning_rate $learning_rate"
    eval $python_command
done
for t in "${task[@]}"
do
    echo "运行task=$t"
    python_command="python ../../traindata12_0.py --state ft --dataset $t --seed 404 --reg 5e-7 --weight_decay $weight_decay --epoch0 1 --epoch $epoch --remain_loss 1 --model $model --target_ratio $target_ratio --pruneFlag up --batchsize $batchsize --learning_rate $learning_rate"
    eval $python_command
done
for t in "${task[@]}"
do
    echo "运行task=$t"
    python_command="python ../../traindata12_0.py --state ft --dataset $t --seed 42 --reg 5e-7 --weight_decay $weight_decay --epoch0 1 --epoch $epoch --remain_loss 1 --model $model --target_ratio $target_ratio --pruneFlag up --batchsize $batchsize --learning_rate $learning_rate"
    eval $python_command
done

epoch=10
batchsize=32
weight_decay=0.002
learning_rate=2e-5
task=('sst2')
for t in "${task[@]}"
do
    echo "运行task=$t"
    python_command="python ../../traindata12_0.py --state ft --dataset $t --seed 3404 --reg 5e-7 --weight_decay $weight_decay --epoch0 1 --epoch $epoch --remain_loss 1 --model $model --target_ratio $target_ratio --pruneFlag up --batchsize $batchsize --learning_rate $learning_rate"
    eval $python_command
done
for t in "${task[@]}"
do
    echo "运行task=$t"
    python_command="python ../../traindata12_0.py --state ft --dataset $t --seed 404 --reg 5e-7 --weight_decay $weight_decay --epoch0 1 --epoch $epoch --remain_loss 1 --model $model --target_ratio $target_ratio --pruneFlag up --batchsize $batchsize --learning_rate $learning_rate"
    eval $python_command
done
for t in "${task[@]}"
do
    echo "运行task=$t"
    python_command="python ../../traindata12_0.py --state ft --dataset $t --seed 42 --reg 5e-7 --weight_decay $weight_decay --epoch0 1 --epoch $epoch --remain_loss 1 --model $model --target_ratio $target_ratio --pruneFlag up --batchsize $batchsize --learning_rate $learning_rate"
    eval $python_command
done