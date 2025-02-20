#bin/bash
export _HF_DEFAULT_ENDPOINT="https://hf-mirror.com"
export HF_ENDPOINT="https://hf-mirror.com"
#python ../../datatest.py
dataset='all'
model="bert-base-uncased"
target_ratio=0.5
method='el2n'

epoch=10
batchsize=32
weight_decay=0.002
learning_rate=2e-5
task=('mrpc' 'rte' 'cola' 'stsb' 'sst2')
for t in "${task[@]}"
do
    echo "运行task=$t"
    python_command="python ../../Single_el2n.py --method $method --state ft --dataset $t --seed 3404 --reg 5e-8 --weight_decay $weight_decay --epoch0 5 --epoch $epoch --remain_loss 0 --model $model --target_ratio $target_ratio --pruneFlag up --batchsize $batchsize --learning_rate $learning_rate"
    eval $python_command
done

epoch=5
batchsize=64
weight_decay=0.002
learning_rate=3e-5
task=('qnli' 'mnli' 'qqp')
for t in "${task[@]}"
do
    echo "运行task=$t"
    python_command="python ../../Single_el2n.py --method $method --state ft --dataset $t --seed 3404 --reg 5e-8 --weight_decay $weight_decay --epoch0 1 --epoch $epoch --remain_loss 0 --model $model --target_ratio $target_ratio --pruneFlag up --batchsize $batchsize --learning_rate $learning_rate"
    eval $python_command
done

epoch=10
batchsize=32
weight_decay=0.002
learning_rate=2e-5
task=('mrpc' 'rte' 'cola' 'stsb' 'sst2')
for t in "${task[@]}"
do
    echo "运行task=$t"
    python_command="python ../../Single_el2n.py --method $method --state ft --dataset $t --seed 42 --reg 5e-8 --weight_decay $weight_decay --epoch0 5 --epoch $epoch --remain_loss 0 --model $model --target_ratio $target_ratio --pruneFlag up --batchsize $batchsize --learning_rate $learning_rate"
    eval $python_command
done

epoch=5
batchsize=64
weight_decay=0.002
learning_rate=3e-5
task=('qnli' 'mnli' 'qqp')
for t in "${task[@]}"
do
    echo "运行task=$t"
    python_command="python ../../Single_el2n.py --method $method --state ft --dataset $t --seed 42 --reg 5e-8 --weight_decay $weight_decay --epoch0 1 --epoch $epoch --remain_loss 0 --model $model --target_ratio $target_ratio --pruneFlag up --batchsize $batchsize --learning_rate $learning_rate"
    eval $python_command
done