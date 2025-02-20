#bin/bash
python ../s1.py --state ft --dataset qnli --seed 3404 --reg 5e-8 --weight_decay 0.0 --epoch 1 --remain_loss 1 --model bert-base-uncased

python ../s1.py --state ft --dataset qnli --seed 3404 --reg 5e-8 --weight_decay 0.0 --epoch 1 --remain_loss 1 --model roberta-base

python ../s1.py --state ft --dataset qnli --seed 3404 --reg 5e-8 --weight_decay 0.0 --epoch 1 --remain_loss 1 --model gpt2

python ../s1.py --state ft --dataset qnli --seed 3404 --reg 5e-8 --weight_decay 0.0 --epoch 1 --remain_loss 1 --model google-t5/t5-base