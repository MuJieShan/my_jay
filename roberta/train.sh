#bin/bash
export _HF_DEFAULT_ENDPOINT=https://hf-mirror.com
export HF_ENDPOINT=https://hf-mirror.com
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

# 定义关联数组
declare -A args

# 解析参数
for i in "$@"; do
    case $i in
        --dataset=*)
            dataset="${i#*=}"
            shift
            ;;
        --seed=*)
            seed="${i#*=}"
            shift
            ;;
        *)
    esac
done
args["dataset"]=$dataset
args["seed"]=$seed
weight_decay=(0.002 0.003 0.004 0.005 0.007)
model="bert-base-uncased"
target_ratio=0.5
epoch=10
batchsize=32
for decay in "${weight_decay[@]}"
do
    echo "运行weight_decay=$decay"
#    python_command="python roberta/train.py --state ft --dataset ${args["dataset"]} --seed ${args["seed"]} --weight_decay $decay --epoch 10 --remain_loss 1"
    python_command="python ../../traindata1.py --state ft --dataset ${args["dataset"]} --seed ${args["seed"]} --reg 5e-8 --weight_decay $decay --epoch $epoch --remain_loss 1 --model $model --target_ratio $target_ratio --pruneFlag up --batchsize $batchsize --optim adamw_torch"
    eval $python_command
    python_command="python ../../traindata1.py --state ft --dataset ${args["dataset"]} --seed ${args["seed"]} --reg 5e-8 --weight_decay $decay --epoch $epoch --remain_loss 1 --model $model --target_ratio $target_ratio --pruneFlag down --batchsize $batchsize --optim adamw_torch"
    eval $python_command
done