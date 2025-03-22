#bin/bash
export _HF_DEFAULT_ENDPOINT="https://hf-mirror.com"
export HF_ENDPOINT="https://hf-mirror.com"
#python ../../traindata_static.py --dataset sst2 --seed 3404 --epoch 10 --epoch0 1 --reg 5e-8 --weight_decay 0 --target_ratio 0.5 --model bert-base-uncased --batchsize 32
#python ../../traindata_static.py --dataset sst2 --seed 3404 --epoch 10 --epoch0 2 --reg 5e-8 --weight_decay 0 --target_ratio 0.5 --model bert-base-uncased --batchsize 32
#python ../../traindata_static1.py --dataset sst2 --seed 3404 --epoch 10 --epoch0 1 --reg 5e-8 --weight_decay 0 --target_ratio 0.5 --model bert-base-uncased --batchsize 32
#python ../../traindata_static1.py --dataset sst2 --seed 3404 --epoch 10 --epoch0 2 --reg 5e-8 --weight_decay 0 --target_ratio 0.5 --model bert-base-uncased --batchsize 32
#python ../../traindata_static2.py --dataset sst2 --seed 3404 --epoch 10 --epoch0 1 --reg 5e-8 --weight_decay 0 --target_ratio 0.5 --model bert-base-uncased --batchsize 32
#python ../../traindata_static2.py --dataset sst2 --seed 3404 --epoch 10 --epoch0 2 --reg 5e-8 --weight_decay 0 --target_ratio 0.5 --model bert-base-uncased --batchsize 32
#python ../../traindata_static.py --dataset mnli --seed 3404 --epoch 10 --epoch0 1 --reg 5e-8 --weight_decay 0 --target_ratio 0.5 --model bert-base-uncased --batchsize 32
#python ../../traindata_static1.py --dataset mnli --seed 3404 --epoch 10 --epoch0 2 --reg 5e-8 --weight_decay 0 --target_ratio 0.5 --model bert-base-uncased --batchsize 32
python ../../traindata_static1.py --dataset mnli --seed 3404 --epoch 10 --epoch0 1 --reg 5e-8 --weight_decay 0 --target_ratio 0.5 --model bert-base-uncased --batchsize 32