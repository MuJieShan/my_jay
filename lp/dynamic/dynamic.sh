#bin/bash
export _HF_DEFAULT_ENDPOINT="https://hf-mirror.com"
export HF_ENDPOINT="https://hf-mirror.com"

python ../../traindata.py --dataset mrpc --seed 3404 --epoch 10 --reg 5e-7 --weight_decay 0.001 --target_ratio 0.5 --model bert-base-uncased --batchsize 32