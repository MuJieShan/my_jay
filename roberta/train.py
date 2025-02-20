import torch
from dataloader import *
from model_loader import get_model_and_tokenizer
from utils import *
from customTrainer import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.data.data_collator import DataCollatorWithPadding
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments,glue_compute_metrics


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
    train_dataset, eval_dataset = get_dataloader(task, model_checkpoint, tokenizer=tokenizer,shuffle=config.shuffle,batch_size=batch_size)
    data_collator = DataCollatorWithPadding(tokenizer)
    # 定义训练参数
    training_args = GlueTrainingArguments(
        state=config.state,
        #training_args
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

        #eval_args
        eval_strategy="epoch",  # "steps",# eval_steps=50,
        save_strategy="no",
        save_steps=1e+6,

        #logging_args
        output_dir=f"./log/{config.dataset}_{config.seed}_{config.weight_decay}_{config.epoch}_{config.state}",
        logging_dir=f"./log/logs/{config.dataset}_{config.seed}_{config.weight_decay}_{config.epoch}_{config.state}",
        logging_steps=50,
        load_best_model_at_end=False,
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
        compute_metrics = lambda eval_pred: compute_metrics(eval_pred, task),
    )
    # 开始训练
    trainer.train()
    if remain_loss:
        loss_history = trainer.get_training_loss()
        loss_file = f"{training_args.output_dir}/loss_history_{training_args.task_name}_{training_args.weight_decay}_{training_args.seed}_{training_args.num_train_epochs}_{training_args.state}.csv"
        df = pd.DataFrame(loss_history)
        df.to_csv(loss_file, index=False)
    end_time = time.time()
    total_time = end_time - start_time
    s = f'Total training time: {total_time}'
    log.info(s)

if __name__ == "__main__":
    main()
#     python roberta/train.py --state pm --dataset sst2 --seed 3404 --weight_decay 0.0 --reg 5e-8 --epoch 1 --remain_loss 0 --model bert-base-uncased
