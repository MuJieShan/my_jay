from transformers.optimization import Adafactor, get_scheduler
from torch.optim import AdamW
from typing import Any, Dict, Union
import math
import torch
import argparse
import numpy as np
from torch import nn
import random
import os
import torch.nn.functional as F
from transformers import glue_compute_metrics
import sys
import json
import pandas as pd
from tqdm import tqdm
import copy
from easydict import EasyDict as edict
from copy import deepcopy
from datasets import Dataset

GLUE_METRIC = {
    'mrpc':'eval_accuracy',
    'sst2':'eval_accuracy',
    'rte':'eval_accuracy',
    'qqp':'eval_accuracy',
    'mnli':'eval_accuracy',
    'qnli':'eval_accuracy',
    'cola':"eval_matthews_correlation",
    'stsb':"eval_pearson",
}
def sort_trainset_by_length(train_dataset,model):
    trainset = deepcopy(train_dataset)
    trainset.reset_format()
    trainset = trainset.map(lambda x: {"length": len(x["input_ids"])})
    trainset = trainset.sort("length")
    if 'roberta' in model or 'gpt' in model:
        columns_to_return = ['label', 'idx', 'input_ids', 'attention_mask']
    else:
        columns_to_return = ['input_ids', 'label', 'attention_mask','token_type_ids']
    trainset.set_format(type='torch', columns=columns_to_return)
    train_dict = {key: train_dataset[key] for key in columns_to_return}
    if 'idx' not in columns_to_return:
        train_dict['idx'] = list(range(len(train_dataset)))
    trainset = Dataset.from_dict(train_dict)
    return trainset
def seed_torch(seed=3404):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def init_log():
    import logging

    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个文件处理器，将日志写入到文件中
    file_handler = logging.FileHandler('example.log')
    file_handler.setLevel(logging.INFO)

    # 创建一个控制台处理器，将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建一个日志记录器，并将处理器添加到记录器中
    logger = logging.getLogger('my_logger')
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
def init_config():
    gpus = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=str, default="ft",help='choose one:[ft,unlabel,pm]')
    parser.add_argument('--dataset', type=str, default="mrpc")
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--epoch0', type=int, default=1)
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=None)
    parser.add_argument('--prune_batchsize', type=int, default=None)
    parser.add_argument('--target_ratio', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--reg', type=float, default=0.5, help="used the reg in state 'unlabel'")
    parser.add_argument('--weight_decay', type=float, default=0.001, help="the coefficient of L2 penalty")
    parser.add_argument('--remain_loss', type=int, default=0)
    parser.add_argument('--shuffle', type=str, default="True")
    parser.add_argument('--pruneFlag', type=str, default="up", help='choose one:[up,down]')
    parser.add_argument('--method', type=str, default="el2n", help='choose one:["gradn", "el2n", "loss"]')
    parser.add_argument('--optim', type=str, default="adamw_torch", help='choose one:["adamw_torch", "sgd", "lion_32bit"]')
    args = parser.parse_args()
    base_config = {'dataset': "mrpc",'model': "bert-base-uncased",'state': "ft",
                   'batchsize': 32, 'epoch': 10,'epoch0': 1,'step':0, 'learning_rate': 2e-5, 'target_ratio': 0.50,
                   'seed': 3404, 'prune_batchsize': 32, 'reg': 0.05, 'weight_decay': 0.001, 'remain_loss': 0, 'shuffle': "True",'pruneFlag': "up",'method': "el2n",'optim': "adamw_torch"}
    config = edict(base_config)
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)
    if config.state == 'pm':
        config.weight_decay = 0.0
        config.shuffle = "False"
    if config.shuffle == "False":
        config.shuffle = False
    else:
        config.shuffle = True
    return config


def prepare_inputs(inputs: Dict[str, Union[torch.Tensor, Any]], device) -> Dict[str, Union[torch.Tensor, Any]]:
    """
    Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs
def compute_loss(model, inputs):
    # if "labels" in inputs:
    #     labels = inputs.pop("labels")
    if "idx" in inputs:
        idx = inputs.pop("idx")
    outputs = model(**inputs)
    if isinstance(outputs, dict) and "loss" not in outputs:
        raise ValueError(
            "The model did not return a loss from the inputs, only the following keys: "
            f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
        )
    # We don't use .loss here since the model may return tuples instead of ModelOutput.
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    return loss
def compute_outputs(model, inputs):
    if "labels" in inputs:
        labels = inputs.pop("labels")
    if "idx" in inputs:
        idx = inputs.pop("idx")
    outputs = model(**inputs,output_hidden_states=True)
    print(outputs.keys())#'logits', 'hidden_states'
    print(outputs['hidden_states'])
    output = outputs['hidden_layer']['pooler_output'].flatten()
    return output

def compute_logits_pooler_output(model, inputs):
    from transformers import BertForSequenceClassification, RobertaForSequenceClassification,GPT2ForSequenceClassification,T5ForSequenceClassification
    if "labels" in inputs:
        labels = inputs.pop("labels")
    if "idx" in inputs:
        idx = inputs.pop("idx")
    outputs = model(**inputs, output_hidden_states=True)
    dropout = None
    dense = None
    tanh = None
    out_proj = None
    pooler_output = None
    # bert
    if isinstance(model, BertForSequenceClassification):
        dropout = model.dropout
        dense = model.bert.pooler.dense
        tanh = torch.tanh
        out_proj = model.classifier

        x = outputs.hidden_states[-1][:, 0, :]
        x = dropout(x)
        x = dense(x)
        pooler_output = tanh(x)
    # roberta
    elif isinstance(model, RobertaForSequenceClassification):
        dropout = model.classifier.dropout
        dense = model.classifier.dense
        tanh = torch.tanh
        out_proj = model.classifier.out_proj

        x = outputs.hidden_states[-1][:, 0, :]
        x = dropout(x)
        x = dense(x)
        pooler_output = tanh(x)
    # T5
    elif isinstance(model, T5ForSequenceClassification):
        dropout = model.classification_head.dropout
        dense = model.classification_head.dense
        tanh = torch.tanh
        out_proj = model.classification_head.out_proj
        x = outputs.decoder_hidden_states[-1][:, -1, :]
        x = dropout(x)
        x = dense(x)
        pooler_output = tanh(x)
    # gpt2
    elif isinstance(model, GPT2ForSequenceClassification):
        pooler_output = outputs.hidden_states[-1][:, -1, :]
    else:
        print("no model")

    return outputs['logits'].flatten(),pooler_output.flatten()

def get_pooler_output(model, inputs):
    from transformers import BertForSequenceClassification, RobertaForSequenceClassification,GPT2ForSequenceClassification,T5ForSequenceClassification
    if "labels" in inputs:
        labels = inputs.pop("labels")
    if "idx" in inputs:
        idx = inputs.pop("idx")
    outputs = model(**inputs, output_hidden_states=True)
    dropout=None
    dense=None
    tanh=None
    out_proj=None
    pooler_output=None
    # bert
    if isinstance(model, BertForSequenceClassification):
        dropout = model.dropout
        dense = model.bert.pooler.dense
        tanh = torch.tanh
        out_proj = model.classifier

        x = outputs.hidden_states[-1][:, 0, :]
        x = dropout(x)
        x = dense(x)
        pooler_output = tanh(x)
    # roberta
    elif isinstance(model, RobertaForSequenceClassification):
        dropout = model.classifier.dropout
        dense = model.classifier.dense
        tanh = torch.tanh
        out_proj = model.classifier.out_proj

        x = outputs.hidden_states[-1][:, 0, :]
        x = dropout(x)
        x = dense(x)
        pooler_output = tanh(x)
    #T5
    elif isinstance(model, T5ForSequenceClassification):
        dropout = model.classification_head.dropout
        dense = model.classification_head.dense
        tanh = torch.tanh
        out_proj = model.classification_head.out_proj
        x = outputs.decoder_hidden_states[-1][:, -1, :]
        x = dropout(x)
        x = dense(x)
        pooler_output = tanh(x)
    # gpt2
    elif isinstance(model, GPT2ForSequenceClassification):
        pooler_output=outputs.hidden_states[-1][:, -1, :]
    else:
        print("no model")

    return pooler_output.flatten()
    # x = dropout(x)
    # x = out_proj(x)
    # print(x)

def statistics_loss1(config, model, train_epoch_iterator, device):
    """
    取分类头上一层的输出，过滤掉分类头对损失颗粒的影响，
    统计直接压缩前后模型输出之差（不同batch）
    压缩一次
    example : !python ../statistics_loss.py --dataset mrpc --seed 3404 --epoch 1 --reg 5e-8 --batchsize 32 --model bert-base-uncased --shuffle False
    """
    loss_before = []
    loss_after = []
    length = len(train_epoch_iterator)
    print('len:', length)
    steps = config.epoch
    iter_num = 0
    compress = config.reg
    for epoch in range(steps):
        iterator = iter(train_epoch_iterator)
        trange = range(len(train_epoch_iterator))
        for step in trange:
            inputs = prepare_inputs(next(iterator), device)
            model.eval()
            step_loss = compute_loss(model, inputs)
            loss_before.append(step_loss.item())
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
            step_loss = compute_loss(model, inputs)
            loss_after.append(step_loss.item())
    loss_gap = [a - b for a, b in zip(loss_after, loss_before)]
    loss_gap_file = f"loss_gap_{config.dataset}_{config.reg}_{config.seed}.csv"
    df = pd.DataFrame(loss_gap)
    df.to_csv(loss_gap_file, index=False)
    print(sum(loss_gap))

def statistics_loss2(config, model, train_epoch_iterator, device):
    """
    取分类头上一层的输出，过滤掉分类头对损失颗粒的影响，
    统计直接压缩前后模型输出之差（不同batch）
    压缩一次
    example : !python ../statistics_loss.py --dataset mrpc --seed 3404 --epoch 1 --reg 5e-8 --batchsize 32 --model bert-base-uncased --shuffle False
    """
    loss_before = []
    loss_after = []
    length = len(train_epoch_iterator)
    print('len:', length)
    steps = config.epoch
    iter_num = 0
    compress = config.reg
    iterator = iter(train_epoch_iterator)
    trange = range(len(train_epoch_iterator))
    for step in trange:
        inputs = prepare_inputs(next(iterator), device)
        model.eval()
        outputs = get_pooler_output(model, inputs)
        outputs = outputs.data.cpu()
        loss_before.append(outputs)
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
        outputs = get_pooler_output(model, inputs)
        outputs = outputs.data.cpu()
        loss_after.append(outputs)
    loss_gap = [(a - b).item() for a, b in zip(loss_after[0], loss_before[0])]
    loss_gap_file = f"pooler_gap_{config.dataset}_{config.reg}_{config.seed}.csv"
    df = pd.DataFrame(loss_gap)
    df.to_csv(loss_gap_file, index=False)
    print(sum(loss_gap))

