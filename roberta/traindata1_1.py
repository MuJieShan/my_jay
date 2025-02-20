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
from safetensors.torch import load_file
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
    print("训练前的pooler_outputs gap start")
    loss_before = []
    loss_after = []
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

    model_checkpoint1 = "/clzs_test003/my_jay/my_jay/roberta/socre/sst2-a/log/sst2/checkpoint-10525"
    model, _ = get_model_and_tokenizer(model_checkpoint1, task, device)
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

    label_file = f"label_file_{config.dataset}_{config.reg}_{config.seed}.csv"
    df = pd.DataFrame(labels_history)
    df.to_csv(label_file, index=False)
    for i in range(len(loss_after)):
        loss_gap = [(a - b).item() for a, b in zip(loss_after[i], loss_before[i])]
        loss_gap_file = f"pooler_gap_{i}_{config.dataset}_{config.reg}_{config.seed}.csv"
        df = pd.DataFrame(loss_gap)
        df.to_csv(loss_gap_file, index=False)
        print(f"norm {i} is {sum([abs(number) for number in loss_gap])}")
    end_time = time.time()
    total_time = end_time - start_time
    s = f'Total training time: {total_time}'
    log.info(s)

if __name__ == "__main__":
    main()

    #保留散度大的样本在微调前后的颗粒状况
    # python ../../traindata1_1.py --dataset mnli --seed 42 --reg 5e-8 --model bert-base-uncased
