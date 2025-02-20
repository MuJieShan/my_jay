#bin/bash
export _HF_DEFAULT_ENDPOINT="https://hf-mirror.com"
export HF_ENDPOINT="https://hf-mirror.com"
dataset='qnli'
model="bert-base-uncased"
target_ratio=0.5
epoch=10
batchsize=32
weight_decay=0

python ../../traindata1.py --state ft --dataset $dataset --seed 3404 --reg 5e-8 --weight_decay $weight_decay --epoch $epoch --remain_loss 1 --model $model --target_ratio 0 --pruneFlag up --batchsize $batchsize --optim adamw_torch

python ../../traindata1.py --state ft --dataset $dataset --seed 3404 --reg 5e-8 --weight_decay $weight_decay --epoch $epoch --remain_loss 1 --model $model --target_ratio $target_ratio --pruneFlag up --batchsize $batchsize --optim adamw_torch

python ../../traindata1.py --state ft --dataset $dataset --seed 3404 --reg 5e-8 --weight_decay $weight_decay --epoch $epoch --remain_loss 1 --model $model --target_ratio $target_ratio --pruneFlag down --batchsize $batchsize --optim adamw_torch

python ../../traindata1.py --state ft --dataset $dataset --seed 42 --reg 5e-8 --weight_decay $weight_decay --epoch $epoch --remain_loss 1 --model $model --target_ratio 0 --pruneFlag up --batchsize $batchsize --optim adamw_torch

python ../../traindata1.py --state ft --dataset $dataset --seed 42 --reg 5e-8 --weight_decay $weight_decay --epoch $epoch --remain_loss 1 --model $model --target_ratio $target_ratio --pruneFlag up --batchsize $batchsize --optim adamw_torch

python ../../traindata1.py --state ft --dataset $dataset --seed 42 --reg 5e-8 --weight_decay $weight_decay --epoch $epoch --remain_loss 1 --model $model --target_ratio $target_ratio --pruneFlag down --batchsize $batchsize --optim adamw_torch

python ../../traindata1.py --state ft --dataset $dataset --seed 404 --reg 5e-8 --weight_decay $weight_decay --epoch $epoch --remain_loss 1 --model $model --target_ratio 0 --pruneFlag up --batchsize $batchsize --optim adamw_torch

python ../../traindata1.py --state ft --dataset $dataset --seed 404 --reg 5e-8 --weight_decay $weight_decay --epoch $epoch --remain_loss 1 --model $model --target_ratio $target_ratio --pruneFlag up --batchsize $batchsize --optim adamw_torch

python ../../traindata1.py --state ft --dataset $dataset --seed 404 --reg 5e-8 --weight_decay $weight_decay --epoch $epoch --remain_loss 1 --model $model --target_ratio $target_ratio --pruneFlag down --batchsize $batchsize --optim adamw_torch
