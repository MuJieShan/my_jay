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
    s = "compress-expand"
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
    model2, _ = get_model_and_tokenizer(model_checkpoint, task, device)
    model2.load_state_dict(copy.deepcopy(model.state_dict()))
    model2.to(next(model.parameters()).device)

    # Load DataLoader
    print(f"\nLoading data...")
    train_dataloader, eval_dataset, trainset = get_dataloader1(task, model_checkpoint, tokenizer=tokenizer,shuffle=config.shuffle,batch_size=batch_size)

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
    test_inputs = prepare_inputs(next(inputdata_iterator), device)
    labels_history = [item.item() for item in test_inputs['labels'].cpu()]
    print("训练前测试")
    model.eval()
    classification_results = np.zeros(32, dtype=int)
    with torch.no_grad():
        if "labels" in test_inputs:
            labels = test_inputs.pop("labels")
        if "idx" in test_inputs:
            idx = test_inputs.pop("idx")
        # 获取模型的预测结果
        outputs = model(**test_inputs)
        logits = outputs.logits  # 获取 logits
        predictions = torch.argmax(logits, dim=1).cpu().numpy()  # 获取预测的类别
    # 比较预测结果和真实标签
    for i in range(32):
        if predictions[i] == labels_history[i]:
            classification_results[i] = 1  # 正确分类
        else:
            classification_results[i] = 0  # 错误分类

    label_file = f"label_0_{config.dataset}.csv"
    df = pd.DataFrame(classification_results)
    df.to_csv(label_file, index=False)
    # compress = config.reg
    compresses = [2e-8,3e-8,4e-8,5e-8,6e-8,7e-8,8e-8,9e-8,1e-7,2e-7,3e-7,4e-7]
    print("训练前的pooler_outputs gap start")
    inputdata = Dataset.from_dict(trainset[:32])
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
    labels_history = [item.item() for item in inputs['labels'].cpu()]

    for compress in compresses:
        model1, _ = get_model_and_tokenizer(model_checkpoint, task, device)
        model1.load_state_dict(copy.deepcopy(model.state_dict()))
        model1.to(next(model.parameters()).device)
        model1.eval()
        outputs = get_pooler_output(model1, inputs)
        outputs = outputs.data.cpu()
        loss_before = outputs
        with torch.no_grad():
            for name, module in model1.named_modules():
                if isinstance(module, torch.nn.Linear):
                    r = 1 + compress
                    module.weight.data = r * module.weight.data
        model1.eval()
        outputs = get_pooler_output(model1, inputs)
        outputs = outputs.data.cpu()
        loss_after = outputs
        loss_gap = [(a - b).item() for a, b in zip(loss_after, loss_before)]
        loss_gap_file = f"pooler_gap_0_{compress}_{config.dataset}.csv"
        df = pd.DataFrame(loss_gap)
        df.to_csv(loss_gap_file, index=False)
    print("训练前的pooler_outputs gap end")

    print("开始训练")
    train_dataset = trainset
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
    model = model2
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
    print("训练后测试")
    model.eval()
    classification_results = np.zeros(32, dtype=int)
    with torch.no_grad():
        if "labels" in test_inputs:
            labels = test_inputs.pop("labels")
        if "idx" in test_inputs:
            idx = test_inputs.pop("idx")
        # 获取模型的预测结果
        outputs = model(**test_inputs)
        logits = outputs.logits  # 获取 logits
        predictions = torch.argmax(logits, dim=1).cpu().numpy()  # 获取预测的类别
    # 比较预测结果和真实标签
    for i in range(32):
        if predictions[i] == labels_history[i]:
            classification_results[i] = 1  # 正确分类
        else:
            classification_results[i] = 0  # 错误分类

    label_file = f"label_1_{config.dataset}.csv"
    df = pd.DataFrame(classification_results)
    df.to_csv(label_file, index=False)
    print("训练后的pooler_outputs gap start")
    for compress in compresses:
        model1, _ = get_model_and_tokenizer(model_checkpoint, task, device)
        model1.load_state_dict(copy.deepcopy(model.state_dict()))
        model1.to(next(model.parameters()).device)
        model1.eval()
        outputs = get_pooler_output(model1, inputs)
        outputs = outputs.data.cpu()
        loss_before = outputs
        with torch.no_grad():
            for name, module in model1.named_modules():
                if isinstance(module, torch.nn.Linear):
                    r = 1 + compress
                    module.weight.data = r * module.weight.data
        model1.eval()
        outputs = get_pooler_output(model1, inputs)
        outputs = outputs.data.cpu()
        loss_after = outputs
        loss_gap = [(a - b).item() for a, b in zip(loss_after, loss_before)]
        loss_gap_file = f"pooler_gap_1_{compress}_{config.dataset}.csv"
        df = pd.DataFrame(loss_gap)
        df.to_csv(loss_gap_file, index=False)
    print("训练后的pooler_outputs gap end")

    label_file = f"label_file_{config.dataset}_{config.seed}_{config.pruneFlag}_{config.epoch0}_{config.target_ratio}_{config.weight_decay}_{config.reg}.csv"
    df = pd.DataFrame(labels_history)
    df.to_csv(label_file, index=False)
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
    #统计训练前后测试集的输出颗粒，模型不同通过数据
    #不同压缩力度下的跳变颗粒
    # python ../../r_123.py --state ft --dataset mrpc --seed 3404 --reg 5e-8 --weight_decay 0.0 --epoch 10 --remain_loss 1 --model bert-base-uncased --target_ratio 0.5 --pruneFlag up --optim adamw_torch --learning_rate 2e-5