#!/bin/bash
python ../train.py --dataset qnli --seed 3404 --epoch 2 --reg 0.001 --reg_1 0.03
python ../train.py --dataset qnli --seed 3404 --epoch 2 --reg 0.001 --reg_1 0.05
