import torch
from dataloader import *
from model_loader import get_model_and_tokenizer
from utils import *
from customTrainer import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.data.data_collator import DataCollatorWithPadding
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments,glue_compute_metrics
from dataPruner import *
import operator
from datasets import Dataset
import time
from callbacks import *
def main():
    start_time = time.time()
    config = init_config()
    log=init_log()
    s = "frozen"
    log.info(s)
    log.info(config)
    seed_torch(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_checkpoint = config.model
    task = config.dataset
    batch_size = config.batchsize
    remain_loss = False
    if config.remain_loss == 1:
        remain_loss = True
    #Load model and tokenizer
    model,tokenizer = get_model_and_tokenizer(model_checkpoint,task,device)
    # Load DataLoader
    print(f"\nLoading data...")
    train_dataloader, eval_dataset, trainset = get_dataloader3(task, model_checkpoint, tokenizer=tokenizer,shuffle=config.shuffle,batch_size=batch_size)
    # data pruning
    compress = config.reg
    data_p = GLUEPruner(dataset=trainset, ratio=config.target_ratio, pruneFlag=config.pruneFlag)
    data_p.prune()
    sampler = data_p.get_sampler()
    train_epoch_iterator = train_dataloader
    print("开始冻结预先训练")
    train_dataset = trainset
    data_collator = DataCollatorWithPadding(tokenizer)
    eval_steps = 10
    # 定义训练参数
    training_args = GlueTrainingArguments(
        state=config.state,
        # training_args
        seed=config.seed,
        learning_rate=config.learning_rate,
        lr_scheduler_type="linear",
        num_train_epochs=config.epoch0,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0,
        # warmup_steps=50,
        weight_decay=config.weight_decay,
        do_train=True,
        reg=config.reg,
        task_name=task,
        shuffle=config.shuffle,
        optim=config.optim,

        # eval_args
        eval_strategy="steps",
        eval_steps=eval_steps,  # "steps","epoch"# eval_steps=50,
        save_strategy="no",
        # save_steps=eval_steps,
        # save_only_model=True,
        # metric_for_best_model=GLUE_METRIC[task],
        # greater_is_better=True,
        # save_safetensors=False,
        # logging_args
        output_dir=f"./log/model/{config.dataset}_{config.seed}_{config.pruneFlag}_{config.target_ratio}_{config.weight_decay}_{config.reg}",
        # load_best_model_at_end=True,
        report_to=["tensorboard"],
    )
    # del model,tokenizer
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    # 创建Trainer实例
    trainer = GlueTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, task),
    )
    trainer.train()
    print("结束冻结预先训练")
    end_time = time.time()
    total_time = end_time - start_time
    s = f'Total training time: {total_time}'
    log.info(s)

if __name__ == "__main__":
    main()
    #只训练前馈头，观察训练过程变化
    # python roberta/train_frozen.py --state ft --dataset sst2 --seed 3404 --pruneFlag up --reg 5e-8 --weight_decay 0.0 --epoch 10 --epoch0 1 --remain_loss 1 --model bert-base-uncased --target_ratio 0.5 --batchsize 32 --optim adamw_torch
