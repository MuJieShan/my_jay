#bin/bash
export _HF_DEFAULT_ENDPOINT="https://hf-mirror.com"
export HF_ENDPOINT="https://hf-mirror.com"
dataset='sst2'
model="bert-base-uncased"
target_ratio=0.5
epoch=5
batchsize=32
weight_decay=0.0

python ../../traindata2.py --state ft --dataset $dataset --seed 404 --reg 5e-8 --weight_decay $weight_decay --epoch $epoch --remain_loss 1 --model $model --target_ratio $target_ratio --pruneFlag up --batchsize $batchsize

python ../../traindata2.py --state ft --dataset $dataset --seed 404 --reg 5e-8 --weight_decay $weight_decay --epoch $epoch --remain_loss 1 --model $model --target_ratio $target_ratio --pruneFlag down --batchsize $batchsize
