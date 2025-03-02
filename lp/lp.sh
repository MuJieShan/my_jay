#bin/bash
#export _HF_DEFAULT_ENDPOINT="https://hf-mirror.com"
#export  ="https://hf-mirror.com"

python ../train.py --dataset mrpc --seed 3404 --epoch 1 --reg 1e-08 --weight_decay 0.001 --model bert-base-uncased --batchsize 32