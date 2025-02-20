#bin/bash
tensorboard --logdir=./logs --load_fast=true
python ../../traindata1.py --state ft --dataset mnli --seed 3404 --reg 5e-8 --weight_decay 0.002 --epoch 5 --remain_loss 1 --model bert-base-uncased --target_ratio 0 --pruneFlag up --batchsize 64 --optim adamw_torch