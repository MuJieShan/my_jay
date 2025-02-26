from transformers.optimization import Adafactor, get_scheduler
from torch.optim import AdamW
from typing import Any, Dict, Union
import math
import torch

import argparse
from easydict import EasyDict as edict
from model import get_metrics, compute_metrics
import numpy as np
from torch import nn
import random
import os
import torch.nn.functional as F
from transformers import glue_compute_metrics
from model import CustomBERTModel, stsb_model
import sys
import json
import pandas as pd
from tqdm import tqdm
import copy
from dataPruner import *
import operator
def get_loss(model, inputs):
    # layer_sums = 0
    # for (name, module) in model.named_modules():
    #     if (pruning.can_prune(module)):
    #         layer_sum = torch.sum(module.weight.pow(2))  # 平方和
    #         #         print(f'{count}:{layer_name}:{layer_sum}')
    #         # r = 0
    #         # r=4e-4
    #         r = 5e-4
    #         # r= reg[name]
    #         layer_sums = layer_sums + 0.5 * r * layer_sum.item()
    if "labels" in inputs:
        labels = inputs.pop("labels")
    if "idx" in inputs:
        idx = inputs.pop("idx")
    outputs = model(**inputs)
    if isinstance(model.loss, torch.nn.MSELoss):
        logits = outputs.logits.squeeze()
        loss = model.loss(logits, labels)  # +layer_sums
    else:
        logits = outputs['logits']
        loss = model.loss(logits, labels)  # +layer_sums

    return loss
def compute_loss(model, inputs):
    # layer_sums = 0
    # for (name, module) in model.named_modules():
    #     if (pruning.can_prune(module)):
    #         layer_sum = torch.sum(module.weight.pow(2))  # 平方和
    #         #         print(f'{count}:{layer_name}:{layer_sum}')
    #         # r = 0
    #         # r=4e-4
    #         r = 5e-4
    #         # r= reg[name]
    #         layer_sums = layer_sums + 0.5 * r * layer_sum.item()
    if "labels" in inputs:
        labels = inputs.pop("labels")
    if "idx" in inputs:
        idx = inputs.pop("idx")
    outputs = model(**inputs)
    if isinstance(model.loss, torch.nn.MSELoss):
        logits = outputs.logits.squeeze()
        loss = model.loss(logits, labels)  # +layer_sums
    else:
        logits = outputs['logits']
        loss = model.loss(logits, labels)  # +layer_sums
    metric, metric_1 = model.compute_metrics(predictions=logits, references=labels)

    return (loss, torch.clone(logits).detach().cpu(), metric, metric_1, torch.clone(labels).detach().cpu())

def compute_loss_roberta(model, inputs):
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

def compute_loss_2(model, inputs):
    # layer_sums = 0
    # for (name, module) in model.named_modules():
    #     if (pruning.can_prune(module)):
    #         layer_sum = torch.sum(module.weight.pow(2))  # 平方和
    #         #         print(f'{count}:{layer_name}:{layer_sum}')
    #         # r = 0
    #         # r=4e-4
    #         r = 5e-4
    #         # r= reg[name]
    #         layer_sums = layer_sums + 0.5 * r * layer_sum.item()
    if "labels" in inputs:
        labels = inputs.pop("labels")
    outputs = model(**inputs)
    if isinstance(model.loss, torch.nn.MSELoss):
        logits = outputs.logits.squeeze()
        loss = model.loss(logits, labels)  # +layer_sums
    else:
        logits = outputs['logits']
        label_logit = torch.mean(logits[np.arange(len(labels)), labels])
        unlabel_logit = torch.mean(logits[np.arange(len(labels)), 1 - labels])
        loss = model.loss(logits, labels)  # +layer_sums
        unlabel_loss = (model.loss(logits, 1 - labels)).item()
    metric, metric_1 = model.compute_metrics(predictions=logits, references=labels)

    return (loss, unlabel_loss,label_logit.item(),unlabel_logit.item(), torch.clone(logits).detach().cpu(), metric, metric_1, torch.clone(labels).detach().cpu())
def compute_loss_3(model, inputs, reg):
    # layer_sums = 0
    # for (name, module) in model.named_modules():
    #     if (pruning.can_prune(module)):
    #         layer_sum = torch.sum(module.weight.pow(2))  # 平方和
    #         #         print(f'{count}:{layer_name}:{layer_sum}')
    #         # r = 0
    #         # r=4e-4
    #         r = 5e-4
    #         # r= reg[name]
    #         layer_sums = layer_sums + 0.5 * r * layer_sum.item()
    if "labels" in inputs:
        labels = inputs.pop("labels")
    outputs = model(**inputs)
    if isinstance(model.loss, torch.nn.MSELoss):
        logits = outputs.logits.squeeze()
        loss = model.loss(logits, labels)  # +layer_sums
    else:
        logits = outputs['logits']
        label_logit = torch.mean(logits[np.arange(len(labels)), labels])
        unlabel_logit = torch.mean(logits[np.arange(len(labels)), 1-labels])

        logsoftmax_func = nn.LogSoftmax(dim=1)
        logsoftmax_logits = logsoftmax_func(logits)
        nllloss_func = nn.NLLLoss()
        label_loss = nllloss_func(logsoftmax_logits, labels)
        unlabel_loss = nllloss_func(logsoftmax_logits, 1 - labels)
        loss = label_loss + 1.0 / (reg * unlabel_loss)
    metric, metric_1 = model.compute_metrics(predictions=logits, references=labels)

    return (loss,label_loss.item(), unlabel_loss.item(), label_logit.item(),unlabel_logit.item(),torch.clone(logits).detach().cpu(), metric, metric_1, torch.clone(labels).detach().cpu())
def compute_loss7(model, inputs,reg):
    if "labels" in inputs:
        labels = inputs.pop("labels")
    outputs = model(**inputs)
    if isinstance(model.loss, torch.nn.MSELoss):
        lossfunction = torch.nn.SmoothL1Loss()
        logits = outputs.logits.squeeze()
        loss = lossfunction(logits,labels)
    else:
        logits = outputs['logits']
        logsoftmax_func = nn.LogSoftmax(dim=1)
        logsoftmax_logits = logsoftmax_func(logits)
        nllloss_func = nn.NLLLoss()
        label_loss = nllloss_func(logsoftmax_logits, labels)
        unlabel_loss = nllloss_func(logsoftmax_logits, 1 - labels)
        loss = label_loss + 1.0/(reg*unlabel_loss)
    metric, metric_1 = model.compute_metrics(predictions=logits, references=labels)

    return (loss, torch.clone(logits).detach().cpu(), metric, metric_1, torch.clone(labels).detach().cpu())

def compute_loss1(model, model1, inputs, pruning):
    """
    """
    # layer_sums = 0
    # for (name, module) in model.named_modules():
    #     if (pruning.can_prune(module)):
    #         layer_sum = torch.sum(module.weight.pow(2))  # 平方和
    #         #         print(f'{count}:{layer_name}:{layer_sum}')
    #         # r = 0
    #         # r=4e-4
    #         r = 5e-4
    #         # r= reg[name]
    #         layer_sums = layer_sums + 0.5 * r * layer_sum.item()
    if "labels" in inputs:
        labels = inputs.pop("labels")
    outputs = model(**inputs)
    outputs1 = model1(**inputs)
    # print(f'logits:{outputs["logits"].shape}')#(bathsize,分类数)

    if isinstance(model.loss, torch.nn.MSELoss):
        logits = outputs.logits.squeeze()
        logits1 = outputs1.logits.squeeze()
        loss = torch.sum(torch.pow(logits - logits1, 2))
        # loss = model.loss(logits, labels)  # +layer_sums
        # loss1 = torch.sum(torch.pow(logits - labels, 2))
        # loss2 = torch.sum(torch.pow(logits - logits1, 2))
        # loss = loss2 if loss2 < loss1 else loss1
    else:
        logits = outputs['logits']
        logits1 = outputs1['logits']
        # loss = model.loss(logits, labels)
        loss = torch.sum(torch.pow(logits - logits1, 2))
    metric, metric_1 = model.compute_metrics(predictions=logits, references=labels)

    return (loss, torch.clone(logits).detach().cpu(), metric, metric_1, torch.clone(labels).detach().cpu())
def compute_loss2(model, model1, inputs, pruning):
    """
    动态对齐损失，好就不动
    """
    if "labels" in inputs:
        labels = inputs.pop("labels")
    outputs = model(**inputs)
    outputs1 = model1(**inputs)
    if isinstance(model.loss, torch.nn.MSELoss):
        logits = outputs.logits.squeeze()
        logits1 = outputs1.logits.squeeze()
        # loss = torch.sum(torch.pow(logits - logits1, 2))
        loss1 = torch.sum(torch.pow(logits - labels, 2))
        loss2 = torch.sum(torch.pow(logits1 - labels, 2))
        if loss2 < loss1:
            loss = torch.sum(torch.pow(logits - logits1, 2))
        else:
            loss = torch.tensor(0)
    else:
        logits = outputs['logits']
        logits1 = outputs1['logits']
        loss1 = model.loss(logits, labels)
        loss2 = model.loss(logits1, labels)
        if loss2 < loss1:
            loss = torch.sum(torch.pow(logits - logits1, 2))
        else:
            loss = torch.tensor(0)
    metric, metric_1 = model.compute_metrics(predictions=logits, references=labels)

    return (loss, torch.clone(logits).detach().cpu(), metric, metric_1, torch.clone(labels).detach().cpu())
def compute_loss3(model, inputs, inputs1, pruning):
    """
    单模型，同类bath损失
    """
    if "labels" in inputs:
        labels = inputs.pop("labels")
    if "labels" in inputs1:
        labels1 = inputs1.pop("labels")
    outputs = model(**inputs)
    outputs1 = model(**inputs1)
    if isinstance(model.loss, torch.nn.MSELoss):
        logits = outputs.logits.squeeze()
        logits1 = outputs1.logits.squeeze()
        loss = torch.sum(torch.pow(logits - logits1, 2))
    else:
        logits = outputs['logits']
        logits1 = outputs1['logits']
        loss = torch.sum(torch.pow(logits - logits1, 2))

    return (loss, torch.clone(logits).detach().cpu())

def compute_loss4(model, model1, inputs, pruning):
    """
    """
    # layer_sums = 0
    # for (name, module) in model.named_modules():
    #     if (pruning.can_prune(module)):
    #         layer_sum = torch.sum(module.weight.pow(2))  # 平方和
    #         #         print(f'{count}:{layer_name}:{layer_sum}')
    #         # r = 0
    #         # r=4e-4
    #         r = 5e-4
    #         # r= reg[name]
    #         layer_sums = layer_sums + 0.5 * r * layer_sum.item()
    if "labels" in inputs:
        labels = inputs.pop("labels")
    outputs = model(**inputs)
    outputs1 = model1(**inputs)
    # print(f'logits:{outputs["logits"].shape}')#(bathsize,分类数)

    if isinstance(model.loss, torch.nn.MSELoss):
        logits = outputs.logits.squeeze()
        logits1 = outputs1.logits.squeeze()
        loss = torch.sum(torch.pow(logits - logits1, 2))
        # loss = model.loss(logits, labels)  # +layer_sums
        # loss1 = torch.sum(torch.pow(logits - labels, 2))
        # loss2 = torch.sum(torch.pow(logits - logits1, 2))
        # loss = loss2 if loss2 < loss1 else loss1
    else:
        logits = outputs['logits']
        logits1 = outputs1['logits']
        # loss = model.loss(logits, labels)
        loss = torch.sum(torch.pow(logits - logits1, 2))

    return (loss, torch.clone(logits).detach().cpu(), torch.clone(labels).detach().cpu())

def compute_loss5(model, inputs, labels, pruning):
    # layer_sums = 0
    # for (name, module) in model.named_modules():
    #     if (pruning.can_prune(module)):
    #         layer_sum = torch.sum(module.weight.pow(2))  # 平方和
    #         #         print(f'{count}:{layer_name}:{layer_sum}')
    #         # r = 0
    #         # r=4e-4
    #         r = 5e-4
    #         # r= reg[name]
    #         layer_sums = layer_sums + 0.5 * r * layer_sum.item()
    outputs = model(**inputs)
    if isinstance(model.loss, torch.nn.MSELoss):
        logits = outputs.logits.squeeze()
        loss = model.loss(logits, labels)  # +layer_sums
    else:
        logits = outputs['logits']
        loss = model.loss(logits, labels)  # +layer_sums
    metric, metric_1 = model.compute_metrics(predictions=logits, references=labels)

    return (loss, torch.clone(logits).detach().cpu(), metric, metric_1, torch.clone(labels).detach().cpu())
def compute_loss6(model, inputs, labels, step,pruning):
    # layer_sums = 0
    # for (name, module) in model.named_modules():
    #     if (pruning.can_prune(module)):
    #         layer_sum = torch.sum(module.weight.pow(2))  # 平方和
    #         #         print(f'{count}:{layer_name}:{layer_sum}')
    #         # r = 0
    #         # r=4e-4
    #         r = 5e-4
    #         # r= reg[name]
    #         layer_sums = layer_sums + 0.5 * r * layer_sum.item()
    outputs = model(**inputs)
    if isinstance(model.loss, torch.nn.MSELoss):
        logits = outputs.logits.squeeze()
        loss = model.loss(logits, labels)  # +layer_sums
    else:
        logits = outputs['logits']
        loss = model.loss(logits, labels)*step/100 # +layer_sums
    metric, metric_1 = model.compute_metrics(predictions=logits, references=labels)

    return (loss, torch.clone(logits).detach().cpu(), metric, metric_1, torch.clone(labels).detach().cpu())

def compute_outputs(model, inputs):
    # layer_sums = 0
    # for (name, module) in model.named_modules():
    #     if (pruning.can_prune(module)):
    #         layer_sum = torch.sum(module.weight.pow(2))  # 平方和
    #         #         print(f'{count}:{layer_name}:{layer_sum}')
    #         # r = 0
    #         # r=4e-4
    #         r = 5e-4
    #         # r= reg[name]
    #         layer_sums = layer_sums + 0.5 * r * layer_sum.item()
    if "labels" in inputs:
        labels = inputs.pop("labels")
    if "idx" in inputs:
        idx = inputs.pop("idx")
    outputs = model(**inputs)
    # print(f'full:{outputs.keys()}')
    # print(f'full:{outputs["hidden_layer"].keys()}')
    # print(f'full:{outputs["hidden_layer"]["pooler_output"].shape}')
    # output = torch.sum(outputs['hidden_layer']['pooler_output'].flatten()).item()
    output = outputs['hidden_layer']['pooler_output'].flatten()
    return output
def compute_output_loss(model,inputs,labels,model1 = None):
    # output = torch.sum(outputs['hidden_layer']['pooler_output'].flatten()).item()
    # output = outputs['hidden_layer']['pooler_output'].flatten()
    # output1 = outputs1['hidden_layer']['pooler_output'].flatten()
    # feature_loss = torch.sum((output1 - output)**2).item()
    if "labels" in inputs:
        labels = inputs.pop("labels")
    outputs = model(**inputs)
    # outputs1 = model1(**inputs)
    logits = outputs['logits']
    # logits1 = outputs1['logits']

    logsoftmax_func = nn.LogSoftmax(dim=1)
    logsoftmax_logits = logsoftmax_func(logits)
    logsoftmax_feature = logsoftmax_func(outputs['hidden_layer']['pooler_output'])

    nllloss_func = nn.NLLLoss()
    label_loss = nllloss_func(logsoftmax_logits, labels).item()
    unlabel_loss = nllloss_func(logsoftmax_logits, 1-labels).item()
    all_loss = torch.mean(-logsoftmax_logits).item()
    feature_loss = torch.mean(-logsoftmax_feature).item()
    loss = model.loss(logits, labels)

    return loss.item(),label_loss,unlabel_loss,all_loss,feature_loss

def compute_output_loss1(model,inputs,labels,model1 = None):
    if "labels" in inputs:
        labels = inputs.pop("labels")
    outputs = model(**inputs)
    outputs1 = model1(**inputs)

    logits = outputs['logits']
    logits1 = outputs1['logits']
    logits_loss = torch.sum((logits1 - logits)**2).item()

    output = outputs['hidden_layer']['pooler_output']
    output1 = outputs1['hidden_layer']['pooler_output']
    feature_loss = torch.sum((output1 - output)**2).item()

    return logits_loss,feature_loss
def eval_compute_loss(model, inputs):
    if "labels" in inputs:
        labels = inputs.pop("labels")
    outputs = model(**inputs)

    if isinstance(model.loss, torch.nn.MSELoss):
        logits = outputs.logits.squeeze()
        loss = model.loss(logits, labels)
    else:
        logits = outputs['logits']
        loss = model.loss(logits, labels)
    metric, metric_1 = model.compute_metrics(predictions=logits, references=labels)

    return (loss, torch.clone(logits).detach().cpu(), metric, metric_1, torch.clone(labels).detach().cpu())


def eval_step(model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    model.eval()
    model.zero_grad()
    loss, _, metric, metric_1, _ = eval_compute_loss(model, inputs)

    return loss.detach(), metric, metric_1


def matthews_correlation(y_true, y_pred):
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)

    tp = np.sum((y_true * y_pred) > 0)
    tn = np.sum((y_true + y_pred) == 0)
    fp = np.sum((y_true < y_pred))
    fn = np.sum((y_true > y_pred))

    numerator = tp * tn - fp * fn
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    if denominator == 0:
        mcc = 0
    else:
        mcc = numerator / denominator

    return mcc


def prepare_inputs(inputs: Dict[str, Union[torch.Tensor, Any]], device) -> Dict[str, Union[torch.Tensor, Any]]:
    """
    Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def create_optimizer(model, adafactor=None, weight_decay=0.0, learning_rate=2e-5,
                     adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, config=None, batch_num=None):
    """
    Setup the optimizer.

    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
    """
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer_cls = Adafactor if adafactor else AdamW
    if adafactor:
        optimizer_cls = Adafactor
        optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
    else:
        optimizer_cls = AdamW
        optimizer_kwargs = {
            "betas": (adam_beta1, adam_beta2),
            "eps": adam_epsilon,

        }
    optimizer_kwargs["lr"] = learning_rate
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer

def create_scheduler(optimizer, lr_scheduler_type: str = "linear", num_training_steps: int = 10,
                     warmup_steps: int = 0, warmup_ratio: float = 0.0):
    """
    Setup the scheduler. The optimizer of the trainer must have been set up before this method is called.

    Args:
        num_training_steps (int): The number of training steps to do.
    """
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=num_training_steps)
    return lr_scheduler


def  load_model(model_checkpoint, task, device):
    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
    if task == 'stsb':
        model = stsb_model(num_labels=num_labels, task=task).to(device)
        for name, param in model.named_modules():
            if name == 'bert.classifier':
                setattr(param, "is_classifier", True)
    else:
        model = CustomBERTModel(model_checkpoint, num_labels=num_labels, task=task).to(device)
    return model


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

def getOneNormofModel(model):
    OneNorm = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                sum = torch.sum(torch.abs(module.weight.data)).item()
                OneNorm += sum
    return  OneNorm


def train_eval_loop(config, model, train_epoch_iterator,eval_epoch_iterator, optimizer, device, log):
    """
    example : !python train.py --dataset mrpc --seed 3404 --epoch 10 --reg 0.001 --reg_1 0.05
    """
    loss_history = []
    OneNormofweight = []
    name1=f"{model.metric.__class__.__name__}"
    train_eval={name1:[]}
    if model.metric_1 != None:
        name2=f'{model.metric_1.__class__.__name__}'
        train_eval[name2] = []
    # Training Loop
    length = len(train_epoch_iterator)
    print('len:', length)
    l = length // 3
    metric_epoch = {}
    steps = config.epoch
    iter_num = 0
    metric_epoch['loss'] = []
    if model.metric != None:
        metric_name = f"{model.metric.__class__.__name__}"
        metric_epoch[f"{model.metric.__class__.__name__}"] = []
    if model.metric_1 != None:
        metric_1_name = f"{model.metric_1.__class__.__name__}"
        metric_epoch[f"{model.metric_1.__class__.__name__}"] = []
    compress = config.reg
    # Eval Loop
    def eval_loop():
        metric_batch_test = {}
        metric_batch_test['loss'] = []
        if model.metric != None:
            metric_batch_test[f"{model.metric.__class__.__name__}"] = []
        if model.metric_1 != None:
            metric_batch_test[f"{model.metric_1.__class__.__name__}"] = []
        if config.dataset == 'stsb' or config.dataset == 'cola':
            trange = range(len(eval_epoch_iterator))
            iterator = iter(eval_epoch_iterator)
            with torch.no_grad():
                model.eval()
                model.zero_grad()
                if config.dataset == 'stsb':
                    ref = np.array([])
                    pre = np.array([])
                else:
                    ref = np.array([], dtype=np.float64)
                    pre = np.array([], dtype=np.float64)
                for step in trange:
                    inputs = prepare_inputs(next(iterator), device)
                    if "labels" in inputs:
                        labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    if config.dataset == 'stsb':
                        predictions = outputs.logits.squeeze()
                    else:
                        predictions = outputs['logits']
                    if config.dataset == 'stsb':
                        ref = np.concatenate((ref, torch.clone(labels).detach().cpu().numpy()), axis=0)
                        pre = np.concatenate((pre, torch.clone(predictions).detach().cpu().numpy()), axis=0)
                    else:
                        for i in range(predictions.shape[0]):
                            pre = np.append(pre, 0 if predictions[i][0] > predictions[i][1] else 1)
                        ref = np.concatenate((ref, torch.clone(labels).detach().cpu().numpy()), axis=0)

                if config.dataset == 'stsb':
                    log.info(str(glue_compute_metrics('sts-b', pre, ref)))
                else:
                    log.info('matthews_correlation:' + str(matthews_correlation(ref, pre)))
        else:
            trange = range(len(eval_epoch_iterator))
            iterator = iter(eval_epoch_iterator)
            with torch.no_grad():
                for step in trange:
                    inputs = prepare_inputs(next(iterator), device)
                    step_loss, step_metric, step_metric_1 = eval_step(model, inputs)
                    metric_batch_test['loss'].append(step_loss.item())
                    if model.metric != None:
                        metric_batch_test[f"{model.metric.__class__.__name__}"].append(
                            list(step_metric.values())[0])
                    if model.metric_1 != None:
                        metric_batch_test[f"{model.metric_1.__class__.__name__}"].append(
                            list(step_metric_1.values())[0])
                    if step == len(eval_epoch_iterator) - 1:
                        log.info('test---')
                        s = f'loss: {sum(metric_batch_test["loss"]) / len(metric_batch_test["loss"])}'
                        if model.metric != None:
                            s += ','
                            s += (
                                f"{model.metric.__class__.__name__}:{sum(metric_batch_test[model.metric.__class__.__name__]) / len(metric_batch_test[model.metric.__class__.__name__])}")
                        if model.metric_1 != None:
                            s += ','
                            s += (
                                f"{model.metric_1.__class__.__name__}: {sum(metric_batch_test[model.metric_1.__class__.__name__]) / len(metric_batch_test[model.metric_1.__class__.__name__])}")
                        log.info(s)

    OneNormofweight.append(getOneNormofModel(model))
    for epoch in range(steps):
        metric_batch = {}
        metric_batch['loss'] = []
        if model.metric != None:
            metric_batch[f"{model.metric.__class__.__name__}"] = []
        if model.metric_1 != None:
            metric_batch[f"{model.metric_1.__class__.__name__}"] = []
        iterator = iter(train_epoch_iterator)
        trange = range(len(train_epoch_iterator))
        for step in trange:
            inputs = prepare_inputs(next(iterator), device)
            model.train()
            optimizer.zero_grad()
            step_loss, logit, step_metric, step_metric_1, _ = compute_loss7(model, inputs, config.reg_1)
            # 惩罚项
            loss_history.append(step_loss.item())
            step_loss.backward()
            train_eval[name1].append(step_metric)
            if step_metric_1:
                train_eval[name2].append(step_metric_1)
            with torch.no_grad():
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        r = compress
                        module.weight.grad += r * module.weight

            optimizer.step()
            OneNormofweight.append(getOneNormofModel(model))
            metric_batch['loss'].append(step_loss.item())
            if model.metric != None:
                metric_batch[f"{model.metric.__class__.__name__}"].append(list(step_metric.values())[0])
            if model.metric_1 != None:
                metric_batch[f"{model.metric_1.__class__.__name__}"].append(list(step_metric_1.values())[0])

            if step % l == 0:
                s = f'train:epoch({epoch})[{step}]/[{length}] lr {optimizer.state_dict()["param_groups"][0]["lr"]} loss {sum(metric_batch["loss"]) / len(metric_batch["loss"])}'
                if model.metric != None:
                    s += ','
                    s += (
                        f"{model.metric.__class__.__name__}: {sum(metric_batch[model.metric.__class__.__name__]) / len(metric_batch[model.metric.__class__.__name__])}")
                if model.metric_1 != None:
                    s += ','
                    s += (
                        f"{model.metric_1.__class__.__name__}: {sum(metric_batch[model.metric_1.__class__.__name__]) / len(metric_batch[model.metric_1.__class__.__name__])}")
                log.info(s)
                eval_loop()
            iter_num += 1
        print(f"********微调epoch{epoch}********")
        eval_loop()
    loss_file = f"loss_{config.dataset}_{config.reg}_{config.reg_1}_{config.seed}.csv"
    df = pd.DataFrame(loss_history)
    df.to_csv(loss_file, index=False)

    name1_file = f"loss_{config.dataset}_{name1}_{config.reg}_{config.reg_1}_{config.seed}.csv"
    df = pd.DataFrame(train_eval[name1])
    df.to_csv(name1_file, index=False)
    if model.metric_1 != None:
        name2_file = f"loss_{config.dataset}_{name2}_{config.reg}_{config.reg_1}_{config.seed}.csv"
        df = pd.DataFrame(train_eval[name2])
        df.to_csv(name2_file, index=False)

    weight_file = f"weight_{config.dataset}_{config.reg}_{config.reg_1}_{config.seed}.csv"
    df = pd.DataFrame(OneNormofweight)
    df.to_csv(weight_file, index=False)
def  train_eval_loop1(config, model, train_epoch_iterator,eval_epoch_iterator, optimizer, device, log):
    """
    example : !python train1.py --dataset mrpc --seed 3404 --epoch 10 --reg 0.001 --reg_1 0.05
    """
    loss_history = []
    label_loss_history = []
    unlabel_loss_history = []
    label_logit_history = []
    unlabel_logit_history = []
    OneNormofweight = []
    # Training Loop
    length = len(train_epoch_iterator)
    print('len:', length)
    l = length // 3
    metric_epoch = {}
    steps = config.epoch
    iter_num = 0
    metric_epoch['loss'] = []
    if model.metric != None:
        metric_name = f"{model.metric.__class__.__name__}"
        metric_epoch[f"{model.metric.__class__.__name__}"] = []
    if model.metric_1 != None:
        metric_1_name = f"{model.metric_1.__class__.__name__}"
        metric_epoch[f"{model.metric_1.__class__.__name__}"] = []
    compress = config.reg
    # Eval Loop
    def eval_loop():
        metric_batch_test = {}
        metric_batch_test['loss'] = []
        if model.metric != None:
            metric_batch_test[f"{model.metric.__class__.__name__}"] = []
        if model.metric_1 != None:
            metric_batch_test[f"{model.metric_1.__class__.__name__}"] = []
        if config.dataset == 'stsb' or config.dataset == 'cola':
            trange = range(len(eval_epoch_iterator))
            iterator = iter(eval_epoch_iterator)
            with torch.no_grad():
                model.eval()
                model.zero_grad()
                if config.dataset == 'stsb':
                    ref = np.array([])
                    pre = np.array([])
                else:
                    ref = np.array([], dtype=np.float64)
                    pre = np.array([], dtype=np.float64)
                for step in trange:
                    inputs = prepare_inputs(next(iterator), device)
                    if "labels" in inputs:
                        labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    if config.dataset == 'stsb':
                        predictions = outputs.logits.squeeze()
                    else:
                        predictions = outputs['logits']
                    if config.dataset == 'stsb':
                        ref = np.concatenate((ref, torch.clone(labels).detach().cpu().numpy()), axis=0)
                        pre = np.concatenate((pre, torch.clone(predictions).detach().cpu().numpy()), axis=0)
                    else:
                        for i in range(predictions.shape[0]):
                            pre = np.append(pre, 0 if predictions[i][0] > predictions[i][1] else 1)
                        ref = np.concatenate((ref, torch.clone(labels).detach().cpu().numpy()), axis=0)

                if config.dataset == 'stsb':
                    log.info(str(glue_compute_metrics('sts-b', pre, ref)))
                else:
                    log.info('matthews_correlation:' + str(matthews_correlation(ref, pre)))
        else:
            trange = range(len(eval_epoch_iterator))
            iterator = iter(eval_epoch_iterator)
            with torch.no_grad():
                for step in trange:
                    inputs = prepare_inputs(next(iterator), device)
                    step_loss, step_metric, step_metric_1 = eval_step(model, inputs)
                    metric_batch_test['loss'].append(step_loss.item())
                    if model.metric != None:
                        metric_batch_test[f"{model.metric.__class__.__name__}"].append(
                            list(step_metric.values())[0])
                    if model.metric_1 != None:
                        metric_batch_test[f"{model.metric_1.__class__.__name__}"].append(
                            list(step_metric_1.values())[0])
                    if step == len(eval_epoch_iterator) - 1:
                        log.info('test---')
                        s = f'loss: {sum(metric_batch_test["loss"]) / len(metric_batch_test["loss"])}'
                        if model.metric != None:
                            s += ','
                            s += (
                                f"{model.metric.__class__.__name__}:{sum(metric_batch_test[model.metric.__class__.__name__]) / len(metric_batch_test[model.metric.__class__.__name__])}")
                        if model.metric_1 != None:
                            s += ','
                            s += (
                                f"{model.metric_1.__class__.__name__}: {sum(metric_batch_test[model.metric_1.__class__.__name__]) / len(metric_batch_test[model.metric_1.__class__.__name__])}")
                        log.info(s)

    OneNormofweight.append(getOneNormofModel(model))
    for epoch in range(steps):
        metric_batch = {}
        metric_batch['loss'] = []
        if model.metric != None:
            metric_batch[f"{model.metric.__class__.__name__}"] = []
        if model.metric_1 != None:
            metric_batch[f"{model.metric_1.__class__.__name__}"] = []
        iterator = iter(train_epoch_iterator)
        trange = range(len(train_epoch_iterator))
        for step in trange:
            inputs = prepare_inputs(next(iterator), device)
            model.train()
            optimizer.zero_grad()
            step_loss,label_loss, unlabel_loss, label_logit,unlabel_logit,logit, step_metric, step_metric_1, _ = compute_loss_3(model, inputs,config.reg_1)
            # 惩罚项
            loss_history.append(step_loss.item())
            label_loss_history.append(label_loss)
            unlabel_loss_history.append(unlabel_loss)
            label_logit_history.append(label_logit)
            unlabel_logit_history.append(unlabel_logit)
            step_loss.backward()

            with torch.no_grad():
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        r = compress
                        module.weight.grad += r * module.weight

            optimizer.step()
            OneNormofweight.append(getOneNormofModel(model))

            metric_batch['loss'].append(step_loss.item())
            if model.metric != None:
                metric_batch[f"{model.metric.__class__.__name__}"].append(list(step_metric.values())[0])
            if model.metric_1 != None:
                metric_batch[f"{model.metric_1.__class__.__name__}"].append(list(step_metric_1.values())[0])

            if step % l == 0:
                s = f'train:epoch({epoch})[{step}]/[{length}] lr {optimizer.state_dict()["param_groups"][0]["lr"]} loss {sum(metric_batch["loss"]) / len(metric_batch["loss"])}'
                if model.metric != None:
                    s += ','
                    s += (
                        f"{model.metric.__class__.__name__}: {sum(metric_batch[model.metric.__class__.__name__]) / len(metric_batch[model.metric.__class__.__name__])}")
                if model.metric_1 != None:
                    s += ','
                    s += (
                        f"{model.metric_1.__class__.__name__}: {sum(metric_batch[model.metric_1.__class__.__name__]) / len(metric_batch[model.metric_1.__class__.__name__])}")
                log.info(s)
                eval_loop()
            iter_num += 1
        print(f"********微调epoch{epoch}********")
        eval_loop()
    loss_file = f"loss_{config.dataset}_{config.reg}_{config.reg_1}_{config.seed}.csv"
    df = pd.DataFrame(loss_history)
    df.to_csv(loss_file, index=False)

    label_loss_file = f"label_loss_{config.dataset}_{config.reg}_{config.reg_1}_{config.seed}.csv"
    df = pd.DataFrame(label_loss_history)
    df.to_csv(label_loss_file, index=False)

    unlabel_loss_file = f"unlabel_loss_{config.dataset}_{config.reg}_{config.reg_1}_{config.seed}.csv"
    df = pd.DataFrame(unlabel_loss_history)
    df.to_csv(unlabel_loss_file, index=False)

    label_logit_file = f"label_logit_{config.dataset}_{config.reg}_{config.reg_1}_{config.seed}.csv"
    df = pd.DataFrame(label_logit_history)
    df.to_csv(label_logit_file, index=False)

    unlabel_logit_file = f"unlabel_logit_{config.dataset}_{config.reg}_{config.reg_1}_{config.seed}.csv"
    df = pd.DataFrame(unlabel_logit_history)
    df.to_csv(unlabel_logit_file, index=False)

    weight_file = f"weight_{config.dataset}_{config.reg}_{config.reg_1}_{config.seed}.csv"
    df = pd.DataFrame(OneNormofweight)
    df.to_csv(weight_file, index=False)
def  train_ft_loop(config, model, train_epoch_iterator,eval_epoch_iterator, optimizer, device, log):
    """
    example : !python train.py --dataset mrpc --seed 3404 --epoch 10 --reg 0.001
    """
    loss_history = []
    OneNormofweight = []
    name1 = f"{model.metric.__class__.__name__}"
    train_eval = {name1: []}
    if model.metric_1 != None:
        name2 = f'{model.metric_1.__class__.__name__}'
        train_eval[name2] = []
    # Training Loop
    length = len(train_epoch_iterator)
    print('len:', length)
    l = length // 3
    metric_epoch = {}
    steps = config.epoch
    iter_num = 0
    metric_epoch['loss'] = []
    if model.metric != None:
        metric_name = f"{model.metric.__class__.__name__}"
        metric_epoch[f"{model.metric.__class__.__name__}"] = []
    if model.metric_1 != None:
        metric_1_name = f"{model.metric_1.__class__.__name__}"
        metric_epoch[f"{model.metric_1.__class__.__name__}"] = []
    compress = config.reg
    # Eval Loop
    def eval_loop():
        metric_batch_test = {}
        metric_batch_test['loss'] = []
        if model.metric != None:
            metric_batch_test[f"{model.metric.__class__.__name__}"] = []
        if model.metric_1 != None:
            metric_batch_test[f"{model.metric_1.__class__.__name__}"] = []
        if config.dataset == 'stsb' or config.dataset == 'cola':
            trange = range(len(eval_epoch_iterator))
            iterator = iter(eval_epoch_iterator)
            with torch.no_grad():
                model.eval()
                model.zero_grad()
                if config.dataset == 'stsb':
                    ref = np.array([])
                    pre = np.array([])
                else:
                    ref = np.array([], dtype=np.float64)
                    pre = np.array([], dtype=np.float64)
                for step in trange:
                    inputs = prepare_inputs(next(iterator), device)
                    if "labels" in inputs:
                        labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    if config.dataset == 'stsb':
                        predictions = outputs.logits.squeeze()
                    else:
                        predictions = outputs['logits']
                    if config.dataset == 'stsb':
                        ref = np.concatenate((ref, torch.clone(labels).detach().cpu().numpy()), axis=0)
                        pre = np.concatenate((pre, torch.clone(predictions).detach().cpu().numpy()), axis=0)
                    else:
                        for i in range(predictions.shape[0]):
                            pre = np.append(pre, 0 if predictions[i][0] > predictions[i][1] else 1)
                        ref = np.concatenate((ref, torch.clone(labels).detach().cpu().numpy()), axis=0)

                if config.dataset == 'stsb':
                    log.info(str(glue_compute_metrics('sts-b', pre, ref)))
                else:
                    log.info('matthews_correlation:' + str(matthews_correlation(ref, pre)))
        else:
            trange = range(len(eval_epoch_iterator))
            iterator = iter(eval_epoch_iterator)
            with torch.no_grad():
                for step in trange:
                    inputs = prepare_inputs(next(iterator), device)
                    step_loss, step_metric, step_metric_1 = eval_step(model, inputs)
                    metric_batch_test['loss'].append(step_loss.item())
                    if model.metric != None:
                        metric_batch_test[f"{model.metric.__class__.__name__}"].append(
                            list(step_metric.values())[0])
                    if model.metric_1 != None:
                        metric_batch_test[f"{model.metric_1.__class__.__name__}"].append(
                            list(step_metric_1.values())[0])
                    if step == len(eval_epoch_iterator) - 1:
                        log.info('test---')
                        s = f'loss: {sum(metric_batch_test["loss"]) / len(metric_batch_test["loss"])}'
                        if model.metric != None:
                            s += ','
                            s += (
                                f"{model.metric.__class__.__name__}:{sum(metric_batch_test[model.metric.__class__.__name__]) / len(metric_batch_test[model.metric.__class__.__name__])}")
                        if model.metric_1 != None:
                            s += ','
                            s += (
                                f"{model.metric_1.__class__.__name__}: {sum(metric_batch_test[model.metric_1.__class__.__name__]) / len(metric_batch_test[model.metric_1.__class__.__name__])}")
                        log.info(s)

    OneNormofweight.append(getOneNormofModel(model))
    for epoch in range(steps):
        metric_batch = {}
        metric_batch['loss'] = []
        if model.metric != None:
            metric_batch[f"{model.metric.__class__.__name__}"] = []
        if model.metric_1 != None:
            metric_batch[f"{model.metric_1.__class__.__name__}"] = []
        iterator = iter(train_epoch_iterator)
        trange = range(len(train_epoch_iterator))
        for step in trange:
            inputs = prepare_inputs(next(iterator), device)
            model.train()
            optimizer.zero_grad()
            step_loss, logit, step_metric, step_metric_1, _ = compute_loss(model, inputs)
            # 惩罚项
            loss_history.append(step_loss.item())
            step_loss.backward()
            train_eval[name1].append(step_metric)
            if step_metric_1:
                train_eval[name2].append(step_metric_1)
            with torch.no_grad():
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        r = compress
                        module.weight.grad += r * module.weight

            optimizer.step()
            OneNormofweight.append(getOneNormofModel(model))

            metric_batch['loss'].append(step_loss.item())
            if model.metric != None:
                metric_batch[f"{model.metric.__class__.__name__}"].append(list(step_metric.values())[0])
            if model.metric_1 != None:
                metric_batch[f"{model.metric_1.__class__.__name__}"].append(list(step_metric_1.values())[0])

            if step % l == 0:
                s = f'train:epoch({epoch})[{step}]/[{length}] lr {optimizer.state_dict()["param_groups"][0]["lr"]} loss {sum(metric_batch["loss"]) / len(metric_batch["loss"])}'
                if model.metric != None:
                    s += ','
                    s += (
                        f"{model.metric.__class__.__name__}: {sum(metric_batch[model.metric.__class__.__name__]) / len(metric_batch[model.metric.__class__.__name__])}")
                if model.metric_1 != None:
                    s += ','
                    s += (
                        f"{model.metric_1.__class__.__name__}: {sum(metric_batch[model.metric_1.__class__.__name__]) / len(metric_batch[model.metric_1.__class__.__name__])}")
                log.info(s)
                eval_loop()
            iter_num += 1
        print(f"********微调epoch{epoch}********")
        eval_loop()
    loss_file = f"loss_ft_{config.dataset}_{config.reg}_{config.seed}.csv"
    df = pd.DataFrame(loss_history)
    df.to_csv(loss_file, index=False)

    name1_file = f"loss_{config.dataset}_{name1}_{config.reg}_{config.reg_1}_{config.seed}.csv"
    df = pd.DataFrame(train_eval[name1])
    df.to_csv(name1_file, index=False)
    if model.metric_1 != None:
        name2_file = f"loss_{config.dataset}_{name2}_{config.reg}_{config.reg_1}_{config.seed}.csv"
        df = pd.DataFrame(train_eval[name2])
        df.to_csv(name2_file, index=False)

    weight_file = f"weight_ft_{config.dataset}_{config.reg}_{config.seed}.csv"
    df = pd.DataFrame(OneNormofweight)
    df.to_csv(weight_file, index=False)
def  train_ft_loop1(config, model, train_epoch_iterator,eval_epoch_iterator, optimizer, device, log):
    """
    example : !python train1.py --dataset mrpc --seed 3404 --epoch 10 --reg 0.001
    """
    loss_history = []
    unlabel_loss_history = []
    label_logit_history = []
    unlabel_logit_history = []
    OneNormofweight = []
    # Training Loop
    length = len(train_epoch_iterator)
    print('len:', length)
    l = length // 2
    metric_epoch = {}
    steps = config.epoch
    iter_num = 0
    metric_epoch['loss'] = []
    if model.metric != None:
        metric_name = f"{model.metric.__class__.__name__}"
        metric_epoch[f"{model.metric.__class__.__name__}"] = []
    if model.metric_1 != None:
        metric_1_name = f"{model.metric_1.__class__.__name__}"
        metric_epoch[f"{model.metric_1.__class__.__name__}"] = []
    compress = config.reg
    # Eval Loop
    def eval_loop():
        metric_batch_test = {}
        metric_batch_test['loss'] = []
        if model.metric != None:
            metric_batch_test[f"{model.metric.__class__.__name__}"] = []
        if model.metric_1 != None:
            metric_batch_test[f"{model.metric_1.__class__.__name__}"] = []
        if config.dataset == 'stsb' or config.dataset == 'cola':
            trange = range(len(eval_epoch_iterator))
            iterator = iter(eval_epoch_iterator)
            with torch.no_grad():
                model.eval()
                model.zero_grad()
                if config.dataset == 'stsb':
                    ref = np.array([])
                    pre = np.array([])
                else:
                    ref = np.array([], dtype=np.float64)
                    pre = np.array([], dtype=np.float64)
                for step in trange:
                    inputs = prepare_inputs(next(iterator), device)
                    if "labels" in inputs:
                        labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    if config.dataset == 'stsb':
                        predictions = outputs.logits.squeeze()
                    else:
                        predictions = outputs['logits']
                    if config.dataset == 'stsb':
                        ref = np.concatenate((ref, torch.clone(labels).detach().cpu().numpy()), axis=0)
                        pre = np.concatenate((pre, torch.clone(predictions).detach().cpu().numpy()), axis=0)
                    else:
                        for i in range(predictions.shape[0]):
                            pre = np.append(pre, 0 if predictions[i][0] > predictions[i][1] else 1)
                        ref = np.concatenate((ref, torch.clone(labels).detach().cpu().numpy()), axis=0)

                if config.dataset == 'stsb':
                    log.info(str(glue_compute_metrics('sts-b', pre, ref)))
                else:
                    log.info('matthews_correlation:' + str(matthews_correlation(ref, pre)))
        else:
            trange = range(len(eval_epoch_iterator))
            iterator = iter(eval_epoch_iterator)
            with torch.no_grad():
                for step in trange:
                    inputs = prepare_inputs(next(iterator), device)
                    step_loss, step_metric, step_metric_1 = eval_step(model, inputs)
                    metric_batch_test['loss'].append(step_loss.item())
                    if model.metric != None:
                        metric_batch_test[f"{model.metric.__class__.__name__}"].append(
                            list(step_metric.values())[0])
                    if model.metric_1 != None:
                        metric_batch_test[f"{model.metric_1.__class__.__name__}"].append(
                            list(step_metric_1.values())[0])
                    if step == len(eval_epoch_iterator) - 1:
                        log.info('test---')
                        s = f'loss: {sum(metric_batch_test["loss"]) / len(metric_batch_test["loss"])}'
                        if model.metric != None:
                            s += ','
                            s += (
                                f"{model.metric.__class__.__name__}:{sum(metric_batch_test[model.metric.__class__.__name__]) / len(metric_batch_test[model.metric.__class__.__name__])}")
                        if model.metric_1 != None:
                            s += ','
                            s += (
                                f"{model.metric_1.__class__.__name__}: {sum(metric_batch_test[model.metric_1.__class__.__name__]) / len(metric_batch_test[model.metric_1.__class__.__name__])}")
                        log.info(s)

    OneNormofweight.append(getOneNormofModel(model))
    for epoch in range(steps):
        metric_batch = {}
        metric_batch['loss'] = []
        if model.metric != None:
            metric_batch[f"{model.metric.__class__.__name__}"] = []
        if model.metric_1 != None:
            metric_batch[f"{model.metric_1.__class__.__name__}"] = []
        iterator = iter(train_epoch_iterator)
        trange = range(len(train_epoch_iterator))
        for step in trange:
            inputs = prepare_inputs(next(iterator), device)
            model.train()
            optimizer.zero_grad()
            step_loss, unlabel_loss, label_logit,unlabel_logit, logit, step_metric, step_metric_1, _ = compute_loss_2(model, inputs)
            # 惩罚项
            loss_history.append(step_loss.item())
            unlabel_loss_history.append(unlabel_loss)
            label_logit_history.append(label_logit)
            unlabel_logit_history.append(unlabel_logit)
            step_loss.backward()

            with torch.no_grad():
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        r = compress
                        module.weight.grad += r * module.weight

            optimizer.step()
            OneNormofweight.append(getOneNormofModel(model))

            metric_batch['loss'].append(step_loss.item())
            if model.metric != None:
                metric_batch[f"{model.metric.__class__.__name__}"].append(list(step_metric.values())[0])
            if model.metric_1 != None:
                metric_batch[f"{model.metric_1.__class__.__name__}"].append(list(step_metric_1.values())[0])

            # if step % l == 0:
            #     s = f'train:epoch({epoch})[{step}]/[{length}] lr {optimizer.state_dict()["param_groups"][0]["lr"]} loss {sum(metric_batch["loss"]) / len(metric_batch["loss"])}'
            #     if model.metric != None:
            #         s += ','
            #         s += (
            #             f"{model.metric.__class__.__name__}: {sum(metric_batch[model.metric.__class__.__name__]) / len(metric_batch[model.metric.__class__.__name__])}")
            #     if model.metric_1 != None:
            #         s += ','
            #         s += (
            #             f"{model.metric_1.__class__.__name__}: {sum(metric_batch[model.metric_1.__class__.__name__]) / len(metric_batch[model.metric_1.__class__.__name__])}")
            #     log.info(s)
            #     eval_loop()
            iter_num += 1
        print(f"********微调epoch{epoch}********")
        eval_loop()
    loss_file = f"loss_ft_{config.dataset}_{config.reg}_{config.seed}.csv"
    df = pd.DataFrame(loss_history)
    df.to_csv(loss_file, index=False)

    unlabel_loss_file = f"unlabel_loss_ft_{config.dataset}_{config.reg}_{config.seed}.csv"
    df = pd.DataFrame(unlabel_loss_history)
    df.to_csv(unlabel_loss_file, index=False)

    label_logit_file = f"label_logit_ft_{config.dataset}_{config.reg}_{config.seed}.csv"
    df = pd.DataFrame(label_logit_history)
    df.to_csv(label_logit_file, index=False)

    unlabel_logit_file = f"unlabel_logit_ft_{config.dataset}_{config.reg}_{config.seed}.csv"
    df = pd.DataFrame(unlabel_logit_history)
    df.to_csv(unlabel_logit_file, index=False)

    weight_file = f"weight_ft_{config.dataset}_{config.reg}_{config.seed}.csv"
    df = pd.DataFrame(OneNormofweight)
    df.to_csv(weight_file, index=False)
#数据剪枝
def  train_ft_loop2(config, model, train_epoch_iterator,eval_epoch_iterator, optimizer, device, log,trainset):
    """
    example : !python traindata.py --dataset mrpc --seed 3404 --epoch 10 --reg 5e-7 --reg2 0.001 --target_ratio 0.5
    """
    loss_history = []
    name1 = f"{model.metric.__class__.__name__}"
    train_eval = {name1: []}
    if model.metric_1 != None:
        name2 = f'{model.metric_1.__class__.__name__}'
        train_eval[name2] = []
    # Training Loop

    metric_epoch = {}
    steps = config.epoch
    iter_num = 0
    metric_epoch['loss'] = []
    if model.metric != None:
        metric_name = f"{model.metric.__class__.__name__}"
        metric_epoch[f"{model.metric.__class__.__name__}"] = []
    if model.metric_1 != None:
        metric_1_name = f"{model.metric_1.__class__.__name__}"
        metric_epoch[f"{model.metric_1.__class__.__name__}"] = []
    compress = config.reg
    compress_ft = config.reg2
    # Eval Loop
    def eval_loop():
        metric_batch_test = {}
        metric_batch_test['loss'] = []
        if model.metric != None:
            metric_batch_test[f"{model.metric.__class__.__name__}"] = []
        if model.metric_1 != None:
            metric_batch_test[f"{model.metric_1.__class__.__name__}"] = []
        if config.dataset == 'stsb' or config.dataset == 'cola':
            trange = range(len(eval_epoch_iterator))
            iterator = iter(eval_epoch_iterator)
            with torch.no_grad():
                model.eval()
                model.zero_grad()
                if config.dataset == 'stsb':
                    ref = np.array([])
                    pre = np.array([])
                else:
                    ref = np.array([], dtype=np.float64)
                    pre = np.array([], dtype=np.float64)
                for step in trange:
                    inputs = prepare_inputs(next(iterator), device)
                    if "labels" in inputs:
                        labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    if config.dataset == 'stsb':
                        predictions = outputs.logits.squeeze()
                    else:
                        predictions = outputs['logits']
                    if config.dataset == 'stsb':
                        ref = np.concatenate((ref, torch.clone(labels).detach().cpu().numpy()), axis=0)
                        pre = np.concatenate((pre, torch.clone(predictions).detach().cpu().numpy()), axis=0)
                    else:
                        for i in range(predictions.shape[0]):
                            pre = np.append(pre, 0 if predictions[i][0] > predictions[i][1] else 1)
                        ref = np.concatenate((ref, torch.clone(labels).detach().cpu().numpy()), axis=0)

                if config.dataset == 'stsb':
                    log.info(str(glue_compute_metrics('sts-b', pre, ref)))
                else:
                    log.info('matthews_correlation:' + str(matthews_correlation(ref, pre)))
        else:
            trange = range(len(eval_epoch_iterator))
            iterator = iter(eval_epoch_iterator)
            with torch.no_grad():
                for step in trange:
                    inputs = prepare_inputs(next(iterator), device)
                    step_loss, step_metric, step_metric_1 = eval_step(model, inputs)
                    metric_batch_test['loss'].append(step_loss.item())
                    if model.metric != None:
                        metric_batch_test[f"{model.metric.__class__.__name__}"].append(
                            list(step_metric.values())[0])
                    if model.metric_1 != None:
                        metric_batch_test[f"{model.metric_1.__class__.__name__}"].append(
                            list(step_metric_1.values())[0])
                    if step == len(eval_epoch_iterator) - 1:
                        log.info('test---')
                        s = f'loss: {sum(metric_batch_test["loss"]) / len(metric_batch_test["loss"])}'
                        if model.metric != None:
                            s += ','
                            s += (
                                f"{model.metric.__class__.__name__}:{sum(metric_batch_test[model.metric.__class__.__name__]) / len(metric_batch_test[model.metric.__class__.__name__])}")
                        if model.metric_1 != None:
                            s += ','
                            s += (
                                f"{model.metric_1.__class__.__name__}: {sum(metric_batch_test[model.metric_1.__class__.__name__]) / len(metric_batch_test[model.metric_1.__class__.__name__])}")
                        log.info(s)

#data pruning
    data_p = GLUEPruner(dataset=trainset, ratio=config.target_ratio)
    data_p.prune()
    sampler = data_p.get_sampler()
    train_epoch_iterator = get_pruned_dataloader(config, trainset,sampler) if config.target_ratio != 0 else train_epoch_iterator
    loss_before={}
    loss_after={}
    iterator = iter(train_epoch_iterator)
    trange = range(len(train_epoch_iterator))
    for step in trange:
        inputs = prepare_inputs(next(iterator), device)
        model.eval()
        step_idx = inputs["idx"]
        outputs = compute_outputs(model, inputs)
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
        outputs = compute_outputs(model, inputs)
        outputs = outputs.reshape(-1, 768)
        assert len(step_idx) == outputs.shape[0], "The length of step_idx must match the number of rows in outputs"
        outputs = outputs.data.cpu()
        for i in range(len(step_idx)):
            loss_after[step_idx[i].item()] = outputs[i]
        del outputs
    loss_gap = {key: torch.max(torch.abs(loss_after[key] - loss_before[key])).item() for key in loss_after if key in loss_before}
    iterator = iter(train_epoch_iterator)
    trange = range(len(train_epoch_iterator))
    for step in trange:
        inputs = prepare_inputs(next(iterator), device)
        # step_size = len(inputs['idx'])
        # step_score = torch.randint(1, 1001, (step_size,))
        get_score = operator.itemgetter(*inputs['idx'].tolist())
        step_score = torch.tensor(get_score(loss_gap))
        data_p.update(step_score, inputs['idx'])
    print(f'修剪前：{len(data_p.cur_index)}')
    data_p.prune()
    print(f'修剪后：{len(data_p.cur_index)}')
    sampler = data_p.get_sampler()
    train_epoch_iterator = get_pruned_dataloader(config, trainset, sampler) if config.target_ratio != 0 else train_epoch_iterator
    length = len(train_epoch_iterator)
    l = length // 3
    print(f"开始训练：数据：{len(train_epoch_iterator)}")
    for epoch in range(steps):
        metric_batch = {}
        metric_batch['loss'] = []
        if model.metric != None:
            metric_batch[f"{model.metric.__class__.__name__}"] = []
        if model.metric_1 != None:
            metric_batch[f"{model.metric_1.__class__.__name__}"] = []
        sampler = data_p.get_sampler()
        train_epoch_iterator = get_pruned_dataloader(config, trainset, sampler) if config.target_ratio != 0 else train_epoch_iterator
        iterator = iter(train_epoch_iterator)
        trange = range(len(train_epoch_iterator))
        for step in trange:
            inputs = prepare_inputs(next(iterator), device)
            model.train()
            optimizer.zero_grad()
            step_loss, logit, step_metric, step_metric_1, _ = compute_loss(model, inputs)
            # 惩罚项
            loss_history.append(step_loss.item())
            step_loss.backward()
            train_eval[name1].append(step_metric)
            if step_metric_1:
                train_eval[name2].append(step_metric_1)
            with torch.no_grad():
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        r = compress_ft
                        module.weight.grad += r * module.weight

            optimizer.step()
            metric_batch['loss'].append(step_loss.item())
            if model.metric != None:
                metric_batch[f"{model.metric.__class__.__name__}"].append(list(step_metric.values())[0])
            if model.metric_1 != None:
                metric_batch[f"{model.metric_1.__class__.__name__}"].append(list(step_metric_1.values())[0])

            # if step % l == 0:
            #     s = f'train:epoch({epoch})[{step}]/[{length}] lr {optimizer.state_dict()["param_groups"][0]["lr"]} loss {sum(metric_batch["loss"]) / len(metric_batch["loss"])}'
            #     if model.metric != None:
            #         s += ','
            #         s += (
            #             f"{model.metric.__class__.__name__}: {sum(metric_batch[model.metric.__class__.__name__]) / len(metric_batch[model.metric.__class__.__name__])}")
            #     if model.metric_1 != None:
            #         s += ','
            #         s += (
            #             f"{model.metric_1.__class__.__name__}: {sum(metric_batch[model.metric_1.__class__.__name__]) / len(metric_batch[model.metric_1.__class__.__name__])}")
            #     log.info(s)
            #     eval_loop()
            iter_num += 1
        print(f"********微调epoch{epoch}********")
        eval_loop()
    # loss_file = f"loss_ft_{config.dataset}_{config.reg}_{config.seed}.csv"
    # df = pd.DataFrame(loss_history)
    # df.to_csv(loss_file, index=False)
    #
    # name1_file = f"loss_{config.dataset}_{name1}_{config.reg}_{config.reg_1}_{config.seed}.csv"
    # df = pd.DataFrame(train_eval[name1])
    # df.to_csv(name1_file, index=False)
    # if model.metric_1 != None:
    #     name2_file = f"loss_{config.dataset}_{name2}_{config.reg}_{config.reg_1}_{config.seed}.csv"
    #     df = pd.DataFrame(train_eval[name2])
    #     df.to_csv(name2_file, index=False)
def statistics_loss_loop(config, model, train_epoch_iterator,eval_epoch_iterator, optimizer, device, log):
    """
    统计直接压缩前后模型输出之差（不同batch）
    压缩一次
    example : !python statistics_loss.py --dataset mrpc --seed 3404 --epoch 1 --reg 5e-7 --reg_1 0.05 --step 0
    """
    loss_before = []
    loss_after = []
    loss_gap = []
    # Training Loop
    length = len(train_epoch_iterator)
    print('len:', length)
    steps = config.epoch
    iter_num = 0

    compress = config.reg

    for epoch in range(steps):
        #不同压缩阶段
        for step_ in range(config.step):
            with torch.no_grad():
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        r = 1 - compress
                        module.weight.data = r * module.weight.data
        #压缩前
        iterator = iter(train_epoch_iterator)
        trange = range(len(train_epoch_iterator))
        # trange = range(1000)

        for step in trange:
            inputs = prepare_inputs(next(iterator), device)
            model.eval()
            # step_loss, _, _, _, _ = compute_loss7(model, inputs, config.reg_1)
            step_loss, logit, step_metric, step_metric_1, labels = compute_loss7(model, inputs, config.reg_1)
            # 压缩前
            loss_before.append(step_loss.item())
            del step_loss,logit,step_metric,step_metric_1,labels
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    r = 1 - compress
                    module.weight.data = r * module.weight.data
        #压缩后
        iterator = iter(train_epoch_iterator)
        trange = range(len(train_epoch_iterator))
        # trange = range(1000)
        for step in trange:
            inputs = prepare_inputs(next(iterator), device)
            model.eval()
            step_loss, logit, step_metric, step_metric_1, labels = compute_loss7(model, inputs, config.reg_1)
            # 压缩后
            loss_after.append(step_loss.item())
            del step_loss,logit,step_metric,step_metric_1,labels

    loss_gap = [a-b for a,b in zip(loss_after,loss_before)]
    loss_gap_file = f"loss_gap_{config.dataset}_{config.reg_1}_{config.reg}_{config.seed}.csv"
    df = pd.DataFrame(loss_gap)
    df.to_csv(loss_gap_file, index=False)
def statistics_ft_loss_loop(config, model, train_epoch_iterator,eval_epoch_iterator, optimizer, device, log):
    """
    统计直接压缩前后模型输出之差（不同batch）
    压缩一次
    example : !python statistics_loss.py --dataset mrpc --seed 3404 --epoch 1 --reg 5e-7 --step 0
    """
    loss_before = []
    loss_after = []
    loss_gap = []
    # Training Loop
    length = len(train_epoch_iterator)
    print('len:', length)
    steps = config.epoch
    iter_num = 0

    compress = config.reg

    for epoch in range(steps):
        #不同压缩阶段
        for step_ in range(config.step):
            with torch.no_grad():
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        r = 1 - compress
                        module.weight.data = r * module.weight.data
        #压缩前
        iterator = iter(train_epoch_iterator)
        trange = range(len(train_epoch_iterator))
        for step in trange:
            inputs = prepare_inputs(next(iterator), device)
            model.eval()
            step_loss, logit, step_metric, step_metric_1, _ = compute_loss(model, inputs)
            # 压缩前
            loss_before.append(step_loss.item())
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    r = 1 - compress
                    module.weight.data = r * module.weight.data
        #压缩后
        iterator = iter(train_epoch_iterator)
        trange = range(len(train_epoch_iterator))
        for step in trange:
            inputs = prepare_inputs(next(iterator), device)
            model.eval()
            step_loss, logit, step_metric, step_metric_1, _ = compute_loss(model, inputs)
            # 压缩后
            loss_after.append(step_loss.item())

    loss_gap = [a-b for a,b in zip(loss_after,loss_before)]
    loss_gap_file = f"ft_loss_gap_{config.dataset}_{config.reg}_{config.seed}.csv"
    df = pd.DataFrame(loss_gap)
    df.to_csv(loss_gap_file, index=False)
    print(loss_before)
    print(loss_gap)
# 取分类头上一层的输出，过滤掉分类头对损失颗粒的影响，统计直接压缩前后模型768维输出之差
def statistics_loss1(config, model, train_epoch_iterator,eval_epoch_iterator, optimizer, device, log):
    """
    取分类头上一层的输出，过滤掉分类头对损失颗粒的影响，
    统计直接压缩前后模型输出之差（不同batch）
    压缩一次
    example : !python statistics_loss.py --dataset mrpc --seed 3404 --epoch 1 --reg 5e-8 --step 0 --batchsize 32 --epoch0 0
    """
    loss_before = []
    loss_after = []
    loss_gap = []
    # Training Loop
    length = len(train_epoch_iterator)
    print('len:', length)
    steps = config.epoch
    iter_num = 0
    compress = config.reg

    for epoch in range(config.epoch0):
        if epoch%3 ==0 :
            for epoch in range(steps):
                # 不同压缩阶段
                for step_ in range(config.step):
                    with torch.no_grad():
                        for name, module in model.named_modules():
                            if isinstance(module, torch.nn.Linear):
                                r = 1 - compress
                                module.weight.data = r * module.weight.data
                # 压缩前
                iterator = iter(train_epoch_iterator)
                # trange = range(len(train_epoch_iterator))
                trange = range(1)
                for step in trange:
                    inputs = prepare_inputs(next(iterator), device)
                    inputs.pop("labels")
                    model.eval()
                    outputs = compute_outputs(model, inputs)
                    outputs = outputs.data.cpu()
                    # outputs = outputs.detach().cpu().numpy()
                    loss_before.append(outputs)
                    del outputs
                with torch.no_grad():
                    for name, module in model.named_modules():
                        if isinstance(module, torch.nn.Linear):
                            r = 1 - compress
                            module.weight.data = r * module.weight.data
                # 压缩后
                iterator = iter(train_epoch_iterator)
                # trange = range(len(train_epoch_iterator))
                trange = range(1)
                for step in trange:
                    inputs = prepare_inputs(next(iterator), device)
                    inputs.pop("labels")
                    model.eval()
                    outputs = compute_outputs(model, inputs)
                    # 压缩后
                    # loss_after.append(outputs.sum())
                    # outputs = outputs.detach().cpu().numpy()
                    outputs = outputs.data.cpu()
                    loss_after.append(outputs)
                    del outputs
        metric_batch = {}
        metric_batch['loss'] = []
        if model.metric != None:
            metric_batch[f"{model.metric.__class__.__name__}"] = []
        if model.metric_1 != None:
            metric_batch[f"{model.metric_1.__class__.__name__}"] = []
        iterator = iter(train_epoch_iterator)
        trange = range(len(train_epoch_iterator))
        for step in trange:
            inputs = prepare_inputs(next(iterator), device)
            model.train()
            optimizer.zero_grad()
            step_loss, logit, step_metric, step_metric_1, _ = compute_loss(model, inputs)
            # 惩罚项
            step_loss.backward()
            optimizer.step()
            metric_batch['loss'].append(step_loss.item())
            if model.metric != None:
                metric_batch[f"{model.metric.__class__.__name__}"].append(list(step_metric.values())[0])
            if model.metric_1 != None:
                metric_batch[f"{model.metric_1.__class__.__name__}"].append(list(step_metric_1.values())[0])

            if step % (length/3) == 0:
                s = f'train:epoch({epoch})[{step}]/[{length}] lr {optimizer.state_dict()["param_groups"][0]["lr"]} loss {sum(metric_batch["loss"]) / len(metric_batch["loss"])}'
                if model.metric != None:
                    s += ','
                    s += (
                        f"{model.metric.__class__.__name__}: {sum(metric_batch[model.metric.__class__.__name__]) / len(metric_batch[model.metric.__class__.__name__])}")
                if model.metric_1 != None:
                    s += ','
                    s += (
                        f"{model.metric_1.__class__.__name__}: {sum(metric_batch[model.metric_1.__class__.__name__]) / len(metric_batch[model.metric_1.__class__.__name__])}")
                log.info(s)
            iter_num += 1
    for epoch in range(steps):
        #不同压缩阶段
        for step_ in range(config.step):
            with torch.no_grad():
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        r = 1 - compress
                        module.weight.data = r * module.weight.data
        #压缩前
        iterator = iter(train_epoch_iterator)
        # trange = range(len(train_epoch_iterator))
        trange = range(1)
        for step in trange:
            inputs = prepare_inputs(next(iterator), device)
            print(inputs)
            inputs.pop("labels")
            model.eval()
            outputs = compute_outputs(model, inputs)
            outputs = outputs.data.cpu()
            # outputs = outputs.detach().cpu().numpy()
            loss_before.append(outputs)
            del outputs
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    r = 1 - compress
                    module.weight.data = r * module.weight.data
        #压缩后
        iterator = iter(train_epoch_iterator)
        # trange = range(len(train_epoch_iterator))
        trange = range(1)
        for step in trange:
            inputs = prepare_inputs(next(iterator), device)
            print(inputs)
            inputs.pop("labels")
            model.eval()
            outputs = compute_outputs(model, inputs)
            # 压缩后
            # loss_after.append(outputs.sum())
            # outputs = outputs.detach().cpu().numpy()
            outputs = outputs.data.cpu()
            loss_after.append(outputs)
            del outputs

    # print(len(loss_before))
    # loss_before_file = f"loss_before_{config.dataset}_{config.reg}_{config.seed}.csv"
    # df = pd.DataFrame(loss_before[0])
    # df.to_csv(loss_before_file, index=False)
    #
    # loss_after_file = f"loss_after_{config.dataset}_{config.reg}_{config.seed}.csv"
    # df = pd.DataFrame(loss_after[0])
    # df.to_csv(loss_after_file, index=False)


    for i in range(len(loss_before)):
        loss_gap = [(a-b).item() for a,b in zip(loss_after[i],loss_before[i])]
        loss_gap_file = f"loss_gap{i}_{config.dataset}_{config.reg}_{config.seed}.csv"
        df = pd.DataFrame(loss_gap)
        df.to_csv(loss_gap_file, index=False)
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
    parser.add_argument('--reg', type=float, default=5e-8, help="used the reg in state 'unlabel'")
    parser.add_argument('--weight_decay', type=float, default=0.001, help="the coefficient of L2 penalty")
    parser.add_argument('--remain_loss', type=int, default=0)
    parser.add_argument('--shuffle', type=str, default="True")
    parser.add_argument('--pruneFlag', type=str, default="up", help='choose one:[up,down]')
    parser.add_argument('--method', type=str, default="el2n", help='choose one:["gradn", "el2n", "loss"]')
    parser.add_argument('--optim', type=str, default="adamw_torch", help='choose one:["adamw_torch", "sgd", "lion_32bit"]')
    args = parser.parse_args()
    base_config = {'dataset': "mrpc",'model': "bert-base-uncased",'state': "ft",
                   'batchsize': 32, 'epoch': 10,'epoch0': 1,'step':0, 'learning_rate': 2e-5, 'target_ratio': 0.50,
                   'seed': 3404, 'prune_batchsize': 32, 'reg': 5e-8, 'weight_decay': 0.001, 'remain_loss': 0, 'shuffle': "True",'pruneFlag': "up",'method': "el2n",'optim': "adamw_torch"}
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


# 每个epoch的每个样本的损失颗粒
def  train_lp_loop(config, model, train_epoch_iterator,train_dataloader1,eval_epoch_iterator, optimizer, device, log):
    """
    计算每个样本的损失颗粒
    每个epoch统计一次
    目的：观察是否真的存在颗粒裂解的情况
    example : !python train.py --dataset mrpc --seed 3404 --epoch 10 --reg 0.001 --model bert-base-uncased --batchsize 32
    """
    name1 = f"{model.metric.__class__.__name__}"
    train_eval = {name1: []}
    if model.metric_1 != None:
        name2 = f'{model.metric_1.__class__.__name__}'
        train_eval[name2] = []
    # Training Loop
    length = len(train_epoch_iterator)
    print('len:', length)
    l = length // 1
    metric_epoch = {}
    steps = config.epoch
    iter_num = 0
    metric_epoch['loss'] = []
    if model.metric != None:
        metric_name = f"{model.metric.__class__.__name__}"
        metric_epoch[f"{model.metric.__class__.__name__}"] = []
    if model.metric_1 != None:
        metric_1_name = f"{model.metric_1.__class__.__name__}"
        metric_epoch[f"{model.metric_1.__class__.__name__}"] = []
    compress = config.reg
    # Eval Loop
    def eval_loop():
        metric_batch_test = {}
        metric_batch_test['loss'] = []
        if model.metric != None:
            metric_batch_test[f"{model.metric.__class__.__name__}"] = []
        if model.metric_1 != None:
            metric_batch_test[f"{model.metric_1.__class__.__name__}"] = []
        if config.dataset == 'stsb' or config.dataset == 'cola':
            trange = range(len(eval_epoch_iterator))
            iterator = iter(eval_epoch_iterator)
            with torch.no_grad():
                model.eval()
                model.zero_grad()
                if config.dataset == 'stsb':
                    ref = np.array([])
                    pre = np.array([])
                else:
                    ref = np.array([], dtype=np.float64)
                    pre = np.array([], dtype=np.float64)
                for step in trange:
                    inputs = prepare_inputs(next(iterator), device)
                    if "labels" in inputs:
                        labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    if config.dataset == 'stsb':
                        predictions = outputs.logits.squeeze()
                    else:
                        predictions = outputs['logits']
                    if config.dataset == 'stsb':
                        ref = np.concatenate((ref, torch.clone(labels).detach().cpu().numpy()), axis=0)
                        pre = np.concatenate((pre, torch.clone(predictions).detach().cpu().numpy()), axis=0)
                    else:
                        for i in range(predictions.shape[0]):
                            pre = np.append(pre, 0 if predictions[i][0] > predictions[i][1] else 1)
                        ref = np.concatenate((ref, torch.clone(labels).detach().cpu().numpy()), axis=0)

                if config.dataset == 'stsb':
                    log.info(str(glue_compute_metrics('sts-b', pre, ref)))
                else:
                    log.info('matthews_correlation:' + str(matthews_correlation(ref, pre)))
        else:
            trange = range(len(eval_epoch_iterator))
            iterator = iter(eval_epoch_iterator)
            with torch.no_grad():
                for step in trange:
                    inputs = prepare_inputs(next(iterator), device)
                    step_loss, step_metric, step_metric_1 = eval_step(model, inputs)
                    metric_batch_test['loss'].append(step_loss.item())
                    if model.metric != None:
                        metric_batch_test[f"{model.metric.__class__.__name__}"].append(
                            list(step_metric.values())[0])
                    if model.metric_1 != None:
                        metric_batch_test[f"{model.metric_1.__class__.__name__}"].append(
                            list(step_metric_1.values())[0])
                    if step == len(eval_epoch_iterator) - 1:
                        log.info('test---')
                        s = f'loss: {sum(metric_batch_test["loss"]) / len(metric_batch_test["loss"])}'
                        if model.metric != None:
                            s += ','
                            s += (
                                f"{model.metric.__class__.__name__}:{sum(metric_batch_test[model.metric.__class__.__name__]) / len(metric_batch_test[model.metric.__class__.__name__])}")
                        if model.metric_1 != None:
                            s += ','
                            s += (
                                f"{model.metric_1.__class__.__name__}: {sum(metric_batch_test[model.metric_1.__class__.__name__]) / len(metric_batch_test[model.metric_1.__class__.__name__])}")
                        log.info(s)


    loss_gap = [[] for _ in range(len(train_dataloader1))]
    def loss_particles(model_lp,e,loss_gap):
        loss_g_before = {}
        iterator = iter(train_dataloader1)
        trange = range(len(train_dataloader1))
        before = tqdm(total=len(train_dataloader1), desc=f"lp before{e}")
        for step in trange:
            before.update(1)
            inputs = prepare_inputs(next(iterator), device)
            model_lp.eval()
            step_idx = inputs["idx"]
            print(inputs.keys())
            loss= get_loss(model_lp, inputs)
            for i in range(len(step_idx)):
                loss_g_before[step_idx[i].item()] = loss.item()
        before.close()

        with torch.no_grad():
            for name, module in model_lp.named_modules():
                if isinstance(module, (torch.nn.Linear)):
                    r = 1 - compress
                    module.weight.data = r * module.weight.data

        loss_g_after = {}
        iterator = iter(train_dataloader1)
        trange = range(len(train_dataloader1))
        after = tqdm(total=len(train_dataloader1), desc=f"lp after{e}")
        for step in trange:
            after.update(1)
            inputs = prepare_inputs(next(iterator), device)
            model_lp.eval()
            step_idx = inputs["idx"]
            loss = get_loss(model_lp, inputs)
            for i in range(len(step_idx)):
                loss_g_after[step_idx[i].item()] = loss.item()
        after.close()

        keys = sorted(loss_g_before.keys())
        loss_g_gap = {key: loss_g_after[key] - loss_g_before[key] for key in keys}
        for key in keys:
            loss_gap[key].append(loss_g_gap[key])
        del model_lp
        # print(loss_gap)

    model_checkpoint = config.model
    task = config.dataset
    for epoch in range(steps):
        model_lp=load_model(model_checkpoint,task,device)
        model_lp.load_state_dict(copy.deepcopy(model.state_dict()))
        model_lp.to(next(model.parameters()).device)
        loss_particles(model_lp, epoch, loss_gap)
        del model_lp
        metric_batch = {}
        metric_batch['loss'] = []
        if model.metric != None:
            metric_batch[f"{model.metric.__class__.__name__}"] = []
        if model.metric_1 != None:
            metric_batch[f"{model.metric_1.__class__.__name__}"] = []
        iterator = iter(train_epoch_iterator)
        trange = range(len(train_epoch_iterator))
        training = tqdm(total=len(train_epoch_iterator), desc=f"training epoch{epoch}")
        for step in trange:
            training.update(1)
            inputs = prepare_inputs(next(iterator), device)
            model.train()
            optimizer.zero_grad()
            step_loss, logit, step_metric, step_metric_1, _ = compute_loss(model, inputs)
            # 惩罚项
            step_loss.backward()
            train_eval[name1].append(step_metric)
            if step_metric_1:
                train_eval[name2].append(step_metric_1)
            with torch.no_grad():
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        r = config.weight_decay
                        module.weight.grad += r * module.weight

            optimizer.step()
            metric_batch['loss'].append(step_loss.item())
            if model.metric != None:
                metric_batch[f"{model.metric.__class__.__name__}"].append(list(step_metric.values())[0])
            if model.metric_1 != None:
                metric_batch[f"{model.metric_1.__class__.__name__}"].append(list(step_metric_1.values())[0])

            if step % l == 0:
                s = f'train:epoch({epoch})[{step}]/[{length}] lr {optimizer.state_dict()["param_groups"][0]["lr"]} loss {sum(metric_batch["loss"]) / len(metric_batch["loss"])}'
                if model.metric != None:
                    s += ','
                    s += (
                        f"{model.metric.__class__.__name__}: {sum(metric_batch[model.metric.__class__.__name__]) / len(metric_batch[model.metric.__class__.__name__])}")
                if model.metric_1 != None:
                    s += ','
                    s += (
                        f"{model.metric_1.__class__.__name__}: {sum(metric_batch[model.metric_1.__class__.__name__]) / len(metric_batch[model.metric_1.__class__.__name__])}")
                log.info(s)
                eval_loop()
            iter_num += 1
        training.close()
        print(f"********微调epoch{epoch}********")
        eval_loop()
    model_lp = load_model(model_checkpoint, task, device)
    model_lp.load_state_dict(copy.deepcopy(model.state_dict()))
    model_lp.to(next(model.parameters()).device)
    loss_particles(model_lp, steps, loss_gap)
    del model_lp

    count_lp = len(loss_gap[0])
    print(count_lp)
    loss_g_file = f"lp_{config['model']}_{config['dataset']}_{config['seed']}_{config['reg']}_{config['batchsize']}.csv"
    df = pd.DataFrame(loss_gap,columns=[i for i in range(count_lp)])
    df.to_csv(loss_g_file, index=False)
    name1_file = f"loss_{config.dataset}_{name1}_{config.reg}_{config.seed}.csv"
    df = pd.DataFrame(train_eval[name1])
    df.to_csv(name1_file, index=False)
    if model.metric_1 != None:
        name2_file = f"loss_{config.dataset}_{name2}_{config.reg}_{config.seed}.csv"
        df = pd.DataFrame(train_eval[name2])
        df.to_csv(name2_file, index=False)

# 不同r、每个样本的损失颗粒
def  train_lp_loop1(config, model, train_epoch_iterator,train_dataloader1,eval_epoch_iterator, optimizer, device, log):
    """
    计算每个样本的损失颗粒
    每个惩罚系数r(2e-8)统计一次
    目的：观察是否真的存在颗粒裂解的情况
    example : !python ../train.py --dataset mrpc --seed 3404 --epoch 1 --reg 5e-08 --weight_decay 0.001 --model bert-base-uncased --batchsize 32
    """
    name1 = f"{model.metric.__class__.__name__}"
    train_eval = {name1: []}
    if model.metric_1 != None:
        name2 = f'{model.metric_1.__class__.__name__}'
        train_eval[name2] = []
    # Training Loop
    length = len(train_epoch_iterator)
    print('len:', length)
    l = length // 1
    metric_epoch = {}
    steps = config.epoch
    iter_num = 0
    metric_epoch['loss'] = []
    if model.metric != None:
        metric_name = f"{model.metric.__class__.__name__}"
        metric_epoch[f"{model.metric.__class__.__name__}"] = []
    if model.metric_1 != None:
        metric_1_name = f"{model.metric_1.__class__.__name__}"
        metric_epoch[f"{model.metric_1.__class__.__name__}"] = []
    compress = config.reg
    # Eval Loop
    def eval_loop():
        metric_batch_test = {}
        metric_batch_test['loss'] = []
        if model.metric != None:
            metric_batch_test[f"{model.metric.__class__.__name__}"] = []
        if model.metric_1 != None:
            metric_batch_test[f"{model.metric_1.__class__.__name__}"] = []
        if config.dataset == 'stsb' or config.dataset == 'cola':
            trange = range(len(eval_epoch_iterator))
            iterator = iter(eval_epoch_iterator)
            with torch.no_grad():
                model.eval()
                model.zero_grad()
                if config.dataset == 'stsb':
                    ref = np.array([])
                    pre = np.array([])
                else:
                    ref = np.array([], dtype=np.float64)
                    pre = np.array([], dtype=np.float64)
                for step in trange:
                    inputs = prepare_inputs(next(iterator), device)
                    if "labels" in inputs:
                        labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    if config.dataset == 'stsb':
                        predictions = outputs.logits.squeeze()
                    else:
                        predictions = outputs['logits']
                    if config.dataset == 'stsb':
                        ref = np.concatenate((ref, torch.clone(labels).detach().cpu().numpy()), axis=0)
                        pre = np.concatenate((pre, torch.clone(predictions).detach().cpu().numpy()), axis=0)
                    else:
                        for i in range(predictions.shape[0]):
                            pre = np.append(pre, 0 if predictions[i][0] > predictions[i][1] else 1)
                        ref = np.concatenate((ref, torch.clone(labels).detach().cpu().numpy()), axis=0)

                if config.dataset == 'stsb':
                    log.info(str(glue_compute_metrics('sts-b', pre, ref)))
                else:
                    log.info('matthews_correlation:' + str(matthews_correlation(ref, pre)))
        else:
            trange = range(len(eval_epoch_iterator))
            iterator = iter(eval_epoch_iterator)
            with torch.no_grad():
                for step in trange:
                    inputs = prepare_inputs(next(iterator), device)
                    step_loss, step_metric, step_metric_1 = eval_step(model, inputs)
                    metric_batch_test['loss'].append(step_loss.item())
                    if model.metric != None:
                        metric_batch_test[f"{model.metric.__class__.__name__}"].append(
                            list(step_metric.values())[0])
                    if model.metric_1 != None:
                        metric_batch_test[f"{model.metric_1.__class__.__name__}"].append(
                            list(step_metric_1.values())[0])
                    if step == len(eval_epoch_iterator) - 1:
                        log.info('test---')
                        s = f'loss: {sum(metric_batch_test["loss"]) / len(metric_batch_test["loss"])}'
                        if model.metric != None:
                            s += ','
                            s += (
                                f"{model.metric.__class__.__name__}:{sum(metric_batch_test[model.metric.__class__.__name__]) / len(metric_batch_test[model.metric.__class__.__name__])}")
                        if model.metric_1 != None:
                            s += ','
                            s += (
                                f"{model.metric_1.__class__.__name__}: {sum(metric_batch_test[model.metric_1.__class__.__name__]) / len(metric_batch_test[model.metric_1.__class__.__name__])}")
                        log.info(s)

    loss_gap = [[] for _ in range(len(train_dataloader1))]
    def loss_particles(model_lp,e,loss_gap):
        loss_g_before = {}
        iterator = iter(train_dataloader1)
        trange = range(len(train_dataloader1))
        before = tqdm(total=len(train_dataloader1), desc=f"lp before{e}")
        for step in trange:
            before.update(1)
            inputs = prepare_inputs(next(iterator), device)
            model_lp.eval()
            step_idx = inputs["idx"]
            # print(inputs.keys())
            loss= get_loss(model_lp, inputs)
            for i in range(len(step_idx)):
                loss_g_before[step_idx[i].item()] = loss.item()
        before.close()

        with torch.no_grad():
            for name, module in model_lp.named_modules():
                if isinstance(module, (torch.nn.Linear)):
                    r = 1 - e
                    module.weight.data = r * module.weight.data

        loss_g_after = {}
        iterator = iter(train_dataloader1)
        trange = range(len(train_dataloader1))
        after = tqdm(total=len(train_dataloader1), desc=f"lp after{e}")
        for step in trange:
            after.update(1)
            inputs = prepare_inputs(next(iterator), device)
            model_lp.eval()
            step_idx = inputs["idx"]
            loss = get_loss(model_lp, inputs)
            for i in range(len(step_idx)):
                loss_g_after[step_idx[i].item()] = loss.item()
        after.close()

        keys = sorted(loss_g_before.keys())
        loss_g_gap = {key: loss_g_after[key] - loss_g_before[key] for key in keys}
        for key in keys:
            loss_gap[key].append(loss_g_gap[key])
        del model_lp
        # print(loss_gap)

    model_checkpoint = config.model
    task = config.dataset
    k_0=1e-08

    # k = [2e-08,3e-08,4e-08,5e-08,6e-08,7e-08,8e-08,9e-08]
    # k = [1e-07,2e-07,3e-07,4e-07,5e-07,6e-07,7e-07,8e-07,9e-07]
    k = [compress+i*k_0 for i in range(0,10)]
    for k_i in k:
        model_lp=load_model(model_checkpoint,task,device)
        model_lp.load_state_dict(copy.deepcopy(model.state_dict()))
        model_lp.to(next(model.parameters()).device)
        loss_particles(model_lp, k_i, loss_gap)
        del model_lp

    count_lp = len(loss_gap[0])
    print(count_lp)
    loss_g_file = f"lp_{config['model']}_{config['dataset']}_{config['seed']}_{config['reg']}_{config['batchsize']}.csv"
    df = pd.DataFrame(loss_gap,columns=[i for i in k])
    df.to_csv(loss_g_file, index=False)
    name1_file = f"loss_{config.dataset}_{name1}_{config.reg}_{config.seed}.csv"
    df = pd.DataFrame(train_eval[name1])
    df.to_csv(name1_file, index=False)
    if model.metric_1 != None:
        name2_file = f"loss_{config.dataset}_{name2}_{config.reg}_{config.seed}.csv"
        df = pd.DataFrame(train_eval[name2])
        df.to_csv(name2_file, index=False)