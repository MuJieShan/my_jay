import numpy as np
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
    seed_torch(3404)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_checkpoint = 'roberta-base'
    # Load DataLoader
    print(f"\nLoading data...")
    task = ['mrpc','rte','cola','stsb','sst2','qnli','mnli','qqp']
    for t in task:
        model, tokenizer = get_model_and_tokenizer(model_checkpoint, t, device)
        train_dataloader, eval_dataset, trainset = get_dataloader1(t, model_checkpoint, tokenizer=tokenizer,shuffle=True,batch_size=32)
        print(len(trainset))
    end_time = time.time()
    total_time = end_time - start_time
    s = f'Total training time: {total_time}'
    log.info(s)

if __name__ == "__main__":
    main()

    #python ../../el2n.py --state ft --dataset mnli --seed 3404 --weight_decay 0.002 --epoch 5 --epoch0 2 --remain_loss 0 --target_ratio 0.5 --model bert-base-uncased --method el2n --pruneFlag up
