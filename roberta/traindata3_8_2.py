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
    train_dataloader, eval_dataset, trainset = get_dataloader1(task, model_checkpoint, tokenizer=tokenizer,shuffle=config.shuffle,batch_size=batch_size)
    # data pruning
    compress = config.reg
    train_epoch_iterator =  train_dataloader
    loss_before = {}
    loss_after = {}
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
    model3.load_state_dict(copy.deepcopy(model.state_dict()))
    model3.to(next(model.parameters()).device)

    train_epoch_iterator = train_dataloader
    loss_before = {}
    loss_after = {}
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
    loss_gap_after = {key: torch.sqrt(torch.sum(torch.square(loss_after[key] - loss_before[key]))).item() for key in loss_after if key in loss_before}
    # loss_gap_after = {key: torch.var(loss_after[key] - loss_before[key]).item() for key in loss_after if key in loss_before}
    # loss_gap_after = {key: torch.max(torch.abs(loss_after[key] - loss_before[key])).item() for key in loss_after if key in loss_before}
    scores={key: abs(loss_gap_after[key]-loss_gap_before[key])*loss_gap_after[key]/loss_gap_before[key] for key in loss_gap_after if key in loss_gap_before}
    import shutil
    shutil.rmtree(training_args.output_dir)
    print(f"Deleted checkpoint file: {training_args.output_dir}")
    data_p = GLUEPruner(dataset=trainset, ratio=config.target_ratio, pruneFlag=config.pruneFlag)
    data_p.prune()
    train_epoch_iterator = train_dataloader
    iterator = iter(train_epoch_iterator)
    trange = range(len(train_epoch_iterator))
    print(len(train_epoch_iterator))
    for step in trange:
        inputs = prepare_inputs(next(iterator), device)
        # step_size = len(inputs['idx'])
        # step_score = torch.randint(1, 1001, (step_size,))
        get_score = operator.itemgetter(*inputs['idx'].tolist())
        step_score = torch.tensor(get_score(scores))
        data_p.update(step_score, inputs['idx'])
    print(f'修剪前：{len(data_p.cur_index)}')
    data_p.prune()
    print(f'修剪后：{len(data_p.cur_index)}')
    data_p.get_scores()

    scores = [v.item() for v in data_p.scores]
    scores_file = f"scores_{config.dataset}_{config.seed}_{config.pruneFlag}_{config.target_ratio}_{config.epoch0}_{config.weight_decay}_{config.reg}.csv"
    df = pd.DataFrame(scores)
    df.to_csv(scores_file, index=False)

    scores = np.array(scores)
    positive_ratio = np.sum(scores > 0) / len(scores)
    negative_ratio = np.sum(scores < 0) / len(scores)

    # 输出结果
    print(f"Positive ratio: {positive_ratio:.4f}")
    print(f"Negative ratio: {negative_ratio:.5f}")
    s=f"Positive ratio: {positive_ratio:.4f}"
    log.info(s)
    s = f"Negative ratio: {negative_ratio:.5f}"
    log.info(s)

    print("训练前的pooler_outputs gap start")
    model = model2
    loss_before = []
    loss_after = []
    # inputdata = data_p.get_largest_score(32)
    inputdata = Dataset.from_dict(eval_dataset[:32])
    data_collator = DataCollatorWithPadding(tokenizer)
    inputdata_dataloader = DataLoader(
        inputdata,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
        # drop_last=dataloader_drop_last,
        num_workers=2,
        pin_memory=True
    )
    inputdata_iterator = iter(inputdata_dataloader)
    inputs = prepare_inputs(next(inputdata_iterator), device)
    labels_history=[item.item() for item in inputs['labels'].cpu()]
    model.eval()
    outputs = get_pooler_output(model, inputs)
    outputs = outputs.data.cpu()
    loss_before.append(outputs)
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                r = 1 - compress
                module.weight.data = r * module.weight.data
    model.eval()
    outputs = get_pooler_output(model, inputs)
    outputs = outputs.data.cpu()
    loss_after.append(outputs)
    print("训练前的pooler_outputs gap end")

    print("开始训练")
    train_dataset = data_p.get_pruned_train_dataset()
    data_collator = DataCollatorWithPadding(tokenizer)
    eval_steps = len(train_epoch_iterator)//3
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
        optim=config.optim,

        #eval_args
        eval_strategy="epoch",
        # eval_steps=eval_steps,# "steps","epoch"# eval_steps=50,
        save_strategy="epoch",
        # save_steps=1e+6,
        save_only_model=True,
        metric_for_best_model=GLUE_METRIC[task],
        greater_is_better=True,
        save_safetensors=False,
        #logging_args
        output_dir=f"./log/model/{config.dataset}_{config.seed}_{config.pruneFlag}_{config.epoch0}_{config.target_ratio}_{config.weight_decay}_{config.reg}",
        logging_dir=f"./log/logs/{config.dataset}_{config.seed}_{config.pruneFlag}_{config.epoch0}_{config.target_ratio}_{config.weight_decay}_{config.reg}",
        logging_steps=50,
        load_best_model_at_end=True,
        report_to=["tensorboard"],
        remain_loss=remain_loss,
    )
    model = model3
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
    print("结束训练")
    s=f'{trainer.evaluate(eval_dataset)}'
    print(s)
    log.info(f'\n{s}\n')
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    import glob,shutil
    checkpoint_files = glob.glob(os.path.join(training_args.output_dir, "checkpoint-*"))
    for file in checkpoint_files:
        shutil.rmtree(file)
        print(f"Deleted checkpoint file: {file}")
    print("训练后的pooler_outputs gap start")
    model.eval()
    outputs = get_pooler_output(model, inputs)
    outputs = outputs.data.cpu()
    loss_before.append(outputs)
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                r = 1 - compress
                module.weight.data = r * module.weight.data
    model.eval()
    outputs = get_pooler_output(model, inputs)
    outputs = outputs.data.cpu()
    loss_after.append(outputs)
    print("训练后的pooler_outputs gap end")

    label_file = f"label_file_{config.dataset}_{config.seed}_{config.pruneFlag}_{config.epoch0}_{config.target_ratio}_{config.weight_decay}_{config.reg}.csv"
    df = pd.DataFrame(labels_history)
    df.to_csv(label_file, index=False)
    for i in range(len(loss_after)):
        loss_gap = [(a - b).item() for a, b in zip(loss_after[i], loss_before[i])]
        norm = sum([abs(number) for number in loss_gap])
        s=f"norm {i} is {norm}"
        print(s)
        log.info(s)
        loss_gap.append(sum([abs(number) for number in loss_gap]))
        loss_gap_file = f"pooler_gap_{i}_{config.dataset}_{config.seed}_{config.pruneFlag}_{config.epoch0}_{config.target_ratio}_{config.weight_decay}_{config.reg}.csv"
        df = pd.DataFrame(loss_gap)
        df.to_csv(loss_gap_file, index=False)
    if remain_loss:
        loss_history = trainer.get_training_loss()
        loss_file = f"{training_args.output_dir}/loss_{training_args.task_name}_{config.epoch0}_{config.target_ratio}_{training_args.weight_decay}_{training_args.seed}_{training_args.num_train_epochs}_{training_args.state}_{config.reg}_{config.pruneFlag}.csv"
        df = pd.DataFrame(loss_history)
        df.to_csv(loss_file, index=False)
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
    # python ../../traindata3.py --state ft --dataset mrpc --seed 3404 --reg 5e-8 --weight_decay 0.0 --epoch0 1 --epoch 10 --remain_loss 1 --model bert-base-uncased --target_ratio 0.5 --pruneFlag up --optim adamw_torch --learning_rate 2e-5