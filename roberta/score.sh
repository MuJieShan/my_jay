#bin/bash
#python ../../scores.py --state ft --dataset mnli --seed 3404 --reg 5e-8 --weight_decay 0.0 --epoch 1 --remain_loss 1 --model bert-base-uncased --target_ratio 0.5 --pruneFlag down
#python ../../scores.py --state ft --dataset mrpc --seed 3404 --reg 5e-8 --weight_decay 0.0 --epoch 1 --remain_loss 1 --model bert-base-uncased --target_ratio 0.5 --pruneFlag down
#python ../../scores.py --state ft --dataset sst2 --seed 3404 --reg 5e-8 --weight_decay 0.0 --epoch 1 --remain_loss 1 --model bert-base-uncased --target_ratio 0.5 --pruneFlag down

python ../../scores.py --state ft --dataset cola --seed 3404 --reg 5e-8 --weight_decay 0.0 --epoch 1 --remain_loss 1 --model bert-base-uncased --target_ratio 0.5 --pruneFlag down

python ../../scores.py --state ft --dataset rte --seed 3404 --reg 5e-8 --weight_decay 0.0 --epoch 1 --remain_loss 1 --model bert-base-uncased --target_ratio 0.5 --pruneFlag down
python ../../scores.py --state ft --dataset qqp --seed 3404 --reg 5e-8 --weight_decay 0.0 --epoch 1 --remain_loss 1 --model bert-base-uncased --target_ratio 0.5 --pruneFlag down