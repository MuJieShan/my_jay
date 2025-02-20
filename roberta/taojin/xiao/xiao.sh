#bin/bash
dataset='all'
model="bert-base-uncased"
target_ratio=0.5

epoch=10
batchsize=32
weight_decay=0.002
learning_rate=2e-5
task=('mrpc' 'rte' 'cola' 'stsb')
for t in "${task[@]}"
do
    echo "运行task=$t"
    python_command="python ../../traindata3_11_1.py --state ft --dataset $t --seed 3404 --reg 1e-7 --weight_decay $weight_decay --epoch0 1 --epoch $epoch --remain_loss 1 --model $model --target_ratio $target_ratio --pruneFlag up --batchsize $batchsize --learning_rate $learning_rate"
    eval $python_command
done
for t in "${task[@]}"
do
    echo "运行task=$t"
    python_command="python ../../traindata3_11_1.py --state ft --dataset $t --seed 3404 --reg 1e-7 --weight_decay $weight_decay --epoch0 1 --epoch $epoch --remain_loss 1 --model $model --target_ratio $target_ratio --pruneFlag down --batchsize $batchsize --learning_rate $learning_rate"
    eval $python_command
done