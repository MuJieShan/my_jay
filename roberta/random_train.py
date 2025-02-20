import torch
from dataloader import get_dataloader,get_dataloader1
from model_loader import get_model_and_tokenizer
from utils import *
from customTrainer import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.data.data_collator import DataCollatorWithPadding
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments,glue_compute_metrics
from dataPruner import *
import operator

import time
def main():
    start_time = time.time()
    config = init_config()
    log=init_log()
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
    train_dataloader, eval_dataset, trainset = get_dataloader1(task, model_checkpoint, tokenizer=tokenizer,shuffle=config.shuffle,batch_size=batch_size)

    # data pruning
    compress = config.reg
    data_p = GLUEPruner(dataset=trainset, ratio=config.target_ratio,pruneFlag=config.pruneFlag)
    data_p.prune()
    print(f'修剪前：{len(data_p.cur_index)}')
    data_p.random_prune()
    print(f'修剪后：{len(data_p.cur_index)}')


    #开始训练
    train_dataset = data_p.get_pruned_train_dataset()
    data_collator = DataCollatorWithPadding(tokenizer)
    eval_steps = len(train_dataloader)//3
    # 定义训练参数
    training_args = GlueTrainingArguments(
        state=config.state,
        # training_args
        seed=config.seed,
        learning_rate=config.learning_rate,
        lr_scheduler_type="linear",
        num_train_epochs=config.epoch,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        # warmup_steps=50,
        weight_decay=config.weight_decay,
        do_train=True,
        reg=config.reg,
        task_name=task,
        shuffle=config.shuffle,
        optim=config.optim,

        # eval_args
        eval_strategy="epoch",
        # eval_steps=eval_steps,# "steps","epoch"# eval_steps=50,
        save_strategy="epoch",
        # save_steps=1e+6,
        save_only_model=True,
        metric_for_best_model=GLUE_METRIC[task],
        greater_is_better=True,
        save_safetensors=False,
        # logging_args
        output_dir=f"./log/model/{config.dataset}_{config.seed}_{config.target_ratio}_{config.weight_decay}_{config.reg}",
        logging_dir=f"./log/logs/{config.dataset}_{config.seed}_{config.target_ratio}_{config.weight_decay}_{config.reg}",
        logging_steps=50,
        load_best_model_at_end=True,
        report_to=["tensorboard"],
        remain_loss=remain_loss,
    )
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
    print("结束训练")
    s = f'{trainer.evaluate(eval_dataset)}'
    print(s)
    log.info(f'\n{s}\n')
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    import glob, shutil
    file = training_args.output_dir
    shutil.rmtree(file)
    print(f"Deleted file: {file}")
    checkpoint_files = glob.glob(os.path.join(training_args.output_dir, "checkpoint-*"))
    if remain_loss:
        loss_history = trainer.get_training_loss()
        loss_file = f"{training_args.output_dir}/loss_{config.dataset}_{config.seed}_{config.target_ratio}_{config.weight_decay}_{config.reg}.csv"
        df = pd.DataFrame(loss_history)
        df.to_csv(loss_file, index=False)
    end_time = time.time()
    total_time = end_time - start_time
    s = f'Total training time: {total_time}'
    log.info(s)

if __name__ == "__main__":
    main()

    #python ../traindata.py --state ft --dataset mrpc --seed 3404 --reg 5e-8 --weight_decay 0.0 --epoch 10 --remain_loss 1 --model bert-base-uncased --target_ratio 0.5
