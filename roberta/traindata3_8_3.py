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
from datasets import Dataset
import time
def main():
    start_time = time.time()
    config = init_config()
    log=init_log()
    s = "|s2-s1|*s2/s1,Euclidean distance"
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
    model1,_=get_model_and_tokenizer(model_checkpoint,task,device)
    model1.load_state_dict(copy.deepcopy(model.state_dict()))
    model1.to(next(model.parameters()).device)
    model2, _ = get_model_and_tokenizer(model_checkpoint, task, device)
    model2.load_state_dict(copy.deepcopy(model.state_dict()))
    model2.to(next(model.parameters()).device)
    # Load DataLoader
    print(f"\nLoading data...")
    train_dataloader, eval_dataset, trainset = get_dataloader1(task, model_checkpoint, tokenizer=tokenizer,shuffle=False,batch_size=batch_size)
    # data pruning
    compress = config.reg
    train_epoch_iterator =  train_dataloader
    loss_before = {}
    loss_after = {}
    iterator = iter(train_epoch_iterator)
    trange = range(len(train_epoch_iterator))
    before = tqdm(total=len(train_epoch_iterator), desc=f"lp before")
    for step in trange:
        before.update(1)
        inputs = prepare_inputs(next(iterator), device)
        model.eval()
        step_idx = inputs["idx"]
        outputs = get_pooler_output(model, inputs)
        outputs = outputs.reshape(-1, 768)
        assert len(step_idx) == outputs.shape[0], "The length of step_idx must match the number of rows in outputs"
        outputs = outputs.data.cpu()
        for i in range(len(step_idx)):
            loss_before[step_idx[i].item()] = outputs[i]
        del outputs
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                r = 1 - compress
                module.weight.data = r * module.weight.data
    iterator = iter(train_epoch_iterator)
    trange = range(len(train_epoch_iterator))
    for step in trange:
        inputs = prepare_inputs(next(iterator), device)
        model.eval()
        step_idx = inputs["idx"]
        outputs = get_pooler_output(model, inputs)
        outputs = outputs.reshape(-1, 768)
        assert len(step_idx) == outputs.shape[0], "The length of step_idx must match the number of rows in outputs"
        outputs = outputs.data.cpu()
        for i in range(len(step_idx)):
            loss_after[step_idx[i].item()] = outputs[i]
        del outputs
    loss_gap_before = {key: torch.sqrt(torch.sum(torch.square(loss_after[key] - loss_before[key]))).item() for key in loss_after if key in loss_before}
    # loss_gap_before = {key: torch.var(loss_after[key] - loss_before[key]).item() for key in loss_after if key in loss_before}
    # loss_gap_before = {key: torch.max(torch.abs(loss_after[key] - loss_before[key])).item() for key in loss_after if key in loss_before}
        # loss_gap = {key: torch.var(loss_after[key] - loss_before[key]).item() for key in loss_after if key in loss_before}


    print("开始预先训练")
    train_dataset = trainset
    data_collator = DataCollatorWithPadding(tokenizer)
    eval_steps = len(train_epoch_iterator) // 2
    # 定义训练参数
    training_args = GlueTrainingArguments(
        state=config.state,
        #training_args
        seed=config.seed,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
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
        optim=config.optim,

        #eval_args
        eval_strategy="steps",
        eval_steps=eval_steps,# "steps","epoch"# eval_steps=50,
        save_strategy="steps",
        save_steps=eval_steps,
        save_only_model=True,
        metric_for_best_model=GLUE_METRIC[task],
        greater_is_better=True,
        save_safetensors=False,
        #logging_args
        output_dir=f"./log/model/{config.dataset}_{config.seed}_{config.pruneFlag}_{config.target_ratio}_{config.weight_decay}_{config.reg}",
        load_best_model_at_end=True,
        report_to=["tensorboard"],
    )
    # del model,tokenizer
    model = model1
    # 创建Trainer实例
    trainer = GlueTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics = lambda eval_pred: compute_metrics(eval_pred, task),
    )
    trainer.train()
    s = f'{trainer.evaluate(eval_dataset)}'
    print(s)
    log.info(f'预先训练{s}')
    print("结束预先训练")
    model3, _ = get_model_and_tokenizer(model_checkpoint, task, device)
    if remain_loss:
        model3.load_state_dict(copy.deepcopy(model.state_dict()))
        model3.to(next(model.parameters()).device)

    train_epoch_iterator = train_dataloader
    loss_before = {}
    loss_after = {}
    losses_before = {}
    losses_after = {}
    iterator = iter(train_epoch_iterator)
    trange = range(len(train_epoch_iterator))
    for step in trange:
        inputs = prepare_inputs(next(iterator), device)
        model.eval()
        step_idx = inputs["idx"]
        with torch.no_grad():
            outputs,losses = get_pooler_output_and_losses(model, inputs)
        outputs = outputs.reshape(-1, 768)
        assert len(step_idx) == outputs.shape[0], "The length of step_idx must match the number of rows in outputs"
        outputs = outputs.data.cpu()
        for i in range(len(step_idx)):
            loss_before[step_idx[i].item()] = outputs[i]
            losses_before[step_idx[i].item()] = losses[i]
        del outputs
    before.close()
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                r = 1 - compress
                module.weight.data = r * module.weight.data
    iterator = iter(train_epoch_iterator)
    trange = range(len(train_epoch_iterator))
    after = tqdm(total=len(train_epoch_iterator), desc=f"lp after")
    for step in trange:
        after.update(1)
        inputs = prepare_inputs(next(iterator), device)
        model.eval()
        step_idx = inputs["idx"]
        with torch.no_grad():
            outputs,losses = get_pooler_output_and_losses(model, inputs)
        outputs = outputs.reshape(-1, 768)
        assert len(step_idx) == outputs.shape[0], "The length of step_idx must match the number of rows in outputs"
        outputs = outputs.data.cpu()
        for i in range(len(step_idx)):
            loss_after[step_idx[i].item()] = outputs[i]
            losses_after[step_idx[i].item()] = losses[i]
        del outputs
    after.close()
    loss_gap_after = {key: torch.sqrt(torch.sum(torch.square(loss_after[key] - loss_before[key]))).item() for key in loss_after if key in loss_before}
    losses_gap = {key: torch.abs(losses_after[key] - losses_before[key]).item() for key in losses_after if key in losses_before}
    # loss_gap_after = {key: torch.var(loss_after[key] - loss_before[key]).item() for key in loss_after if key in loss_before}
    # loss_gap_after = {key: torch.max(torch.abs(loss_after[key] - loss_before[key])).item() for key in loss_after if key in loss_before}
    scores={key: abs(loss_gap_after[key]-loss_gap_before[key])*loss_gap_after[key]/loss_gap_before[key] for key in loss_gap_after if key in loss_gap_before}
    output_dir = f"./log/model/{config.dataset}_{config.seed}_{config.pruneFlag}_{config.target_ratio}_{config.weight_decay}_{config.reg}"
    pooler_file = f"{output_dir}_pooler_gap.pt"
    torch.save(scores, pooler_file)
    # pooler_gap = torch.load(pooler_file)
    losses_file = f"{output_dir}_losses_gap.pt"
    torch.save(losses_gap, losses_file)
    losses_gap = torch.load(losses_file)
    import shutil
    shutil.rmtree(training_args.output_dir)
    print(f"Deleted checkpoint file: {training_args.output_dir}")

    end_time = time.time()
    total_time = end_time - start_time
    s = f'Total training time: {total_time}'
    log.info(s)


if __name__ == "__main__":
    main()
    # 剪枝标准：
        # Euclidean distance
        # |s2-s1|*s2/s1
    # 测试集样本在微调前后的颗粒状况
    #先统计每个样本微调前后的散度（得分)
    # python roberta/traindata3_8_2.py --state ft --dataset sst2 --seed 3404 --reg 5e-8 --weight_decay 0.002 --epoch0 1 --epoch 10 --remain_loss 1 --model roberta-base --target_ratio 0.5 --pruneFlag up --optim adamw_torch --learning_rate 2e-5 --batchsize 32