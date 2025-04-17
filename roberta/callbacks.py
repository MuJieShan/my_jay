from copy import deepcopy
from functools import partial
import time

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from transformers import TrainerCallback
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback
import copy
import csv
def compute_grad(output, parameters, loss_attr: str = "loss"):
    grads = torch.autograd.grad(getattr(output, loss_attr), parameters)
    grads = torch.concat([torch.reshape(g.detach().cpu(), (-1,)) for g in grads])
    return torch.norm(grads)


def compute_sample_grads(inputs, model, device):
    """ manually process each sample with per sample gradient """
    batch_size = inputs["input_ids"].shape[0]
    loss_attrs = "loss"
    sample_grads = []
    for i in range(batch_size):
        batch_input = {k: inputs[k][i].unsqueeze(0).to(device) for k in inputs}
        output = model(**batch_input)
        sample_grads.append(compute_grad(output, model.classifier.parameters(), loss_attrs))
        del batch_input
    return sample_grads


def compute_el2n(inputs, model, device):
    batch_input = {k: inputs[k].to(device) for k in inputs}
    with torch.no_grad():
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            output = model(**batch_input)

    logits_attr = "logits"
    label_key = "labels"
    num_attr = "num_labels"

    num_classes = getattr(model, num_attr)
    if num_classes > 1:
        p = nn.functional.softmax(getattr(output, logits_attr), dim=1)
        y = nn.functional.one_hot(batch_input[label_key], num_classes=num_classes)
    else:
        p = nn.functional.sigmoid(getattr(output, logits_attr))
        y = batch_input[label_key].unsqueeze(dim=1)
    err = p - y
    scores = torch.norm(err, dim=1)
    return scores.detach().cpu()

def compute_persample_loss(inputs, model, device):
    batch_input = {k: inputs[k].to(device) for k in inputs}
    with torch.no_grad():
        # device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device, dtype=torch.float16):
            output = model(**batch_input)

    logits_attr = "logits"
    label_key = "labels"
    num_attr = "num_labels"

    logits = getattr(output, logits_attr)
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    label = nn.functional.one_hot(batch_input[label_key], num_classes=getattr(model, num_attr))
    loss = loss_fct(logits.float(), label.float())

    return loss.detach().cpu()


class ScoreCallback(TrainerCallback):
    def __init__(self, method: str, dataset, collate_fn, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.scores = {}
        self._set_compute_fn()
        self.weight = []

    def _set_compute_fn(self):
        if self.method == "loss":
            self.compute_fn = partial(compute_persample_loss)
        elif self.method == "el2n":
            self.compute_fn = partial(compute_el2n)
        elif self.method == "gradn":
            self.compute_fn = compute_sample_grads

    def prepare_inputs(self,inputs, device):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        return inputs
    def on_train_end(self, args, state, control, model=None, **kwargs):
        print(f"SCORE EVAL: {self.method}")
        model.to(args.device)
        if self.method == "el2n":
            batch_size = 32
        elif self.method == "gradn":
            batch_size = 12
        else:
            batch_size = args.per_device_eval_batch_size
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=args.shuffle
        )
        all_grads = {}
        iterator = iter(dataloader)
        trange = range(len(dataloader))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for step in trange:
            batch = self.prepare_inputs(next(iterator), device)
            if "idx" in batch:
                idx = batch.pop("idx")
            batch_grads = self.compute_fn(inputs=batch, model=model, device=args.device)
            for i, n in zip(idx, batch_grads):
                all_grads[i.item()] = n.item()
        self.scores = all_grads


class LossCallback(TrainerCallback):
    """
    General total loss compute per eval.
    """
    def on_evaluate(self, args, state, control, model=None, train_dataloader=None, **kwargs):
        print("COMPUTING TRAIN LOSS")
        model.to(args.device)
        all_loss = []
        count = 0
        for inputs in tqdm(train_dataloader):
            batch_input = {k: inputs[k].to(args.device) for k in inputs}
            with torch.no_grad():
                output = model(**batch_input)
            bs = int(batch_input["input_ids"].size(dim=0))
            all_loss.append(float(output.loss.detach().cpu()) * bs)
            count += bs

        with open(f"{args.output_dir}/loss.tsv", "a") as fp:
            fp.write("%f\t%f\n" % (state.epoch, sum(all_loss)/count))


class TimeCallback(TrainerCallback):
    """
    Compute time per step (along process logging in time.txt)
    and summing at the end of the training.
    """
    def __init__(self):
        self.start_time = 0
        self.total_time = 0
    def on_step_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
    def on_step_end(self, args, state, control, **kwargs):
        step_time = time.time() - self.start_time
        with open(f"{args.output_dir}/time.txt", "a") as fp:
            fp.write("%f\t%f\tstep\n" % (state.global_step, step_time))
    def on_train_end(self, args, state, control, **kwargs):
        with open(f"{args.output_dir}/time.txt", "r") as fp:
            times = [tuple(f.strip().split("\t")) for f in fp.readlines()]
        self.total_time = sum([float(t) for _, t, _ in times])


class WeightCallback(TrainerCallback):
    """
    General model weight norm per epoch.
    """
    def on_epoch_end(self, args, state, control,model=None, **kwargs):
        print("COMPUTING WEIGHT NORM")
        model.to(args.device)
        WeightNorm = 0
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    sum = torch.sum(torch.abs(module.weight.data)).item()
                    WeightNorm += sum
        with open(f"{args.output_dir}/weight.tsv", "a") as fp:
            fp.write("%f\t%f\n" % (state.epoch, WeightNorm))


class WeightNormCallback(TrainerCallback):
    def __init__(self, tb_writer,initmodel):
        self.tb_writer = tb_writer
        self.old_model = copy.deepcopy(initmodel)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = state.epoch
        # before = 0.0
        # with torch.no_grad():
        #     for name, module in self.old_model.named_modules():
        #         if "classifier" not in name and isinstance(module, torch.nn.Linear):
        #             sum = torch.norm(module.weight.data, p=1).item()
        #             # sum = torch.norm(module.weight.data, p=1).item()
        #             before += sum
        # with open(f"{args.output_dir}/weightbefore.tsv", "a") as fp:
        #     fp.write("%f\t%f\n" % (state.epoch, before))

        FroNorm = 0.0
        with torch.no_grad():
            for name, old_module in self.old_model.named_modules():
                if "classifier" not in name and isinstance(old_module, torch.nn.Linear):
                    new_module = model.get_submodule(name)
                    for old_param, new_param in zip(old_module.parameters(), new_module.parameters()):
                        param_diff = new_param - old_param
                        sum = torch.norm(param_diff, p=1).item()
                        FroNorm += sum
        with open(f"{args.output_dir}/weight1norm.csv", "a", newline='') as fp:
            csv_writer = csv.writer(fp)
            csv_writer.writerow([state.epoch, FroNorm])
        self.old_model.load_state_dict(model.state_dict())
        # after = 0.0
        # with torch.no_grad():
        #     for name, module in model.named_modules():
        #         if "classifier" not in name and isinstance(module, torch.nn.Linear):
        #             sum = torch.norm(module.weight.data, p=1).item()
        #             # sum = torch.norm(module.weight.data, p=1).item()
        #             after += sum
        # gap = after - before
        # with open(f"{args.output_dir}/gap.tsv", "a") as fp:
        #     fp.write("%f\t%f\n" % (state.epoch, gap))
        self.tb_writer.add_scalar('Weight Norm', FroNorm, epoch)
        self.tb_writer.flush()
        torch.cuda.empty_cache()

def getWeightNormCallback(log_dir='',initmodel=None):
    tb_writer = SummaryWriter(log_dir=log_dir)
    return WeightNormCallback(tb_writer,initmodel)




