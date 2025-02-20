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
from callbacks import *
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
    print(len(trainset))
    #pre-finetuning
    print("开始训练")
    total_scores=[]
    seed = config.seed
    seed_torch(seed)
    # model, _ = get_model_and_tokenizer(model_checkpoint, task, device)
    train_dataset = copy.deepcopy(trainset)
    data_collator = DataCollatorWithPadding(tokenizer)
    # 定义训练参数
    training_args = GlueTrainingArguments(
        state=config.state,
        # training_args
        seed=seed,
        learning_rate=config.learning_rate,
        lr_scheduler_type="linear",
        num_train_epochs=config.epoch0,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        # warmup_steps=50,
        weight_decay=config.weight_decay,
        do_train=True,
        reg=config.reg,
        task_name=task,
        shuffle=config.shuffle,

        # eval_args
        eval_strategy="epoch",
        # eval_steps=eval_steps,# "steps","epoch"# eval_steps=50,
        save_strategy="no",
        save_steps=1e+8,
        output_dir=f"./log/pre/{config.dataset}",
        report_to=["tensorboard"],
        load_best_model_at_end=False,
    )
    trainer = GlueTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, task),
    )
    if config.method in ["gradn", "el2n", "loss"]:
        # Reduce padding and runtime in forward pass by sorting.
        score_callback = ScoreCallback(
            method=config.method,
            dataset=trainset,
            collate_fn=data_collator,
        )
    trainer.add_callback(score_callback)
    trainer.train()
    print("结束训练")
    #updata importance scores of data
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, ScoreCallback):
            print("获取分数")
            total_scores.append(callback.scores)
            print(sum(callback.scores.values()))
            break
    print("score average")
    from collections import defaultdict
    scores = defaultdict(list)
    for d in total_scores:
        for key, value in d.items():
            scores[key].append(value)
    averaged_scores = {key: sum(values) / len(values) for key, values in scores.items()}
    print(sum(averaged_scores.values()))
    data_p = GLUEPruner(dataset=trainset, ratio=config.target_ratio, pruneFlag=config.pruneFlag)
    data_p.prune()
    sampler = data_p.get_sampler()
    train_epoch_iterator = get_pruned_dataloader(config, trainset, sampler) if config.target_ratio != 0 else train_dataloader
    iterator = iter(train_epoch_iterator)
    trange = range(len(train_epoch_iterator))
    for step in trange:
        inputs = prepare_inputs(next(iterator), device)
        get_score = operator.itemgetter(*inputs['idx'].tolist())
        step_score = torch.tensor(get_score(averaged_scores))
        data_p.update(step_score, inputs['idx'])
    print(f'修剪前：{len(data_p.cur_index)}')
    data_p.prune()
    print(f'修剪后：{len(data_p.cur_index)}')
    # 开始训练
    train_dataset = data_p.get_pruned_train_dataset()
    # model, tokenizer = get_model_and_tokenizer(model_checkpoint, task, device)
    model.zero_grad()
    data_collator = DataCollatorWithPadding(tokenizer)
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

    import shutil
    file = training_args.output_dir
    shutil.rmtree(file)
    print(f"Deleted file: {file}")
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

    #python ../../el2n.py --state ft --dataset mnli --seed 3404 --weight_decay 0.002 --epoch 5 --epoch0 2 --remain_loss 0 --target_ratio 0.5 --model bert-base-uncased --method el2n --pruneFlag up
