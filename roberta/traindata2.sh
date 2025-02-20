#bin/bash
export _HF_DEFAULT_ENDPOINT="https://hf-mirror.com"
export HF_ENDPOINT="https://hf-mirror.com"
dataset='mrpc'
model="bert-base-uncased"
target_ratio=0.5
epoch=10
batchsize=32
weight_decay=0.002
learning_rate=2e-5

epoch0=(3 4 5 6 7 8 9 10)
for e in "${epoch0[@]}"
do
    echo "运行epoch0=$e"
    python_command="python ../../traindata2.py --state ft --dataset $dataset --seed 3404 --reg 5e-8 --weight_decay $weight_decay --epoch0 $e --epoch $epoch --remain_loss 1 --model $model --target_ratio $target_ratio --pruneFlag up --batchsize $batchsize --learning_rate $learning_rate"
    eval $python_command
    python_command="python ../../traindata2.py --state ft --dataset $dataset --seed 3404 --reg 5e-8 --weight_decay $weight_decay --epoch0 $e --epoch $epoch --remain_loss 1 --model $model --target_ratio $target_ratio --pruneFlag down --batchsize $batchsize --learning_rate $learning_rate"
    eval $python_command
done