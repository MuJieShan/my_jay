#bin/bash
export _HF_DEFAULT_ENDPOINT="https://hf-mirror.com"
export HF_ENDPOINT="https://hf-mirror.com"
dataset='all'
model="roberta-base"
target_ratio=0.5

epoch=5
batchsize=64
weight_decay=0.01
learning_rate=2e-5
task='qnli'

echo "运行task=$t"
python_command="python ../../less.py --state ft --dataset $task --seed 3404 --reg $r --weight_decay $weight_decay --epoch0 1 --epoch $epoch --remain_loss 1 --model $model --target_ratio $target_ratio --pruneFlag up --batchsize $batchsize --learning_rate $learning_rate"
eval $python_command
