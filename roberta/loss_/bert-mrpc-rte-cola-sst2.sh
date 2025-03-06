#bin/bash
#损失颗粒
#python roberta/train.py --state pm --dataset mrpc --seed 3404 --reg 5e-7 --weight_decay 0.001 --epoch 1
#python roberta/train.py --state pm --dataset cola --seed 3404 --reg 5e-7 --weight_decay 0.001 --epoch 1
#python roberta/train.py --state pm --dataset rte --seed 3404 --reg 5e-7 --weight_decay 0.001 --epoch 1
#python roberta/train.py --state pm --dataset sst2 --seed 3404 --reg 5e-7 --weight_decay 0.001 --epoch 1
#python roberta/train.py --state pm --dataset wnli --seed 3404 --reg 5e-7 --weight_decay 0.001 --epoch 1

#微调
#python roberta/train.py --state ft --dataset mrpc --seed 3404 --weight_decay 0.0 --epoch 10 --remain_loss 1
#python roberta/train.py --state ft --dataset cola --seed 3404 --weight_decay 0.0 --epoch 10 --remain_loss 1
#python roberta/train.py --state ft --dataset rte --seed 3404 --weight_decay 0.0 --epoch 10 --remain_loss 1
#python roberta/train.py --state ft --dataset sst2 --seed 3404 --weight_decay 0.0 --epoch 10 --remain_loss 1
#python roberta/train.py --state ft --dataset wnli --seed 3404 --weight_decay 0.0 --epoch 10 --remain_loss 1

dataset=('mrpc' 'rte' 'cola' 'stsb')
model="bert-base-uncased"
epoch=10
batchsize=32
for task in "${dataset[@]}"
do
    echo "运行task=$task"
    python_command="python roberta/train.py --state ft --dataset $task --seed 3404 --weight_decay 0 --epoch $epoch --remain_loss 1"
    eval $python_command
    python_command="python roberta/train.py --state ft --dataset $task --seed 3404 --weight_decay 0.002 --epoch $epoch --remain_loss 1"
    eval $python_command
done