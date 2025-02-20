import math
import time

from tqdm.auto import tqdm
from transformers import (Trainer, TrainingArguments, EvalPrediction)
from transformers.training_args import OptimizerNames
from evaluate import load
import evaluate
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers.utils import is_datasets_available
from transformers.trainer_utils import (seed_worker,
                                        enable_full_determinism,
                                        find_executable_batch_size,
                                        set_seed)
import warnings
from transformers.pytorch_utils import  Conv1D
state=["ft","unlabel","pm"]

def compute_metrics(eval_pred, task):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    metric = load("glue", task)
    if  task== "stsb":
        results = metric.compute(predictions=predictions, references=labels)
    else:
        predictions = np.argmax(predictions, axis=1)
        results = metric.compute(predictions=predictions, references=labels)
    return results

class GlueTrainingArguments(TrainingArguments):
    def __init__(self,task_name: str = '', state: str = '', reg: int = 1, shuffle:bool =True, remain_loss:bool = False,**kwargs):
        super().__init__(**kwargs)
        self.task_name = task_name
        self.state = state
        self.reg = reg
        self.shuffle = shuffle
        self.remain_loss = remain_loss

class GlueTrainer(Trainer):
    def __init__(self, compute_loss_func = None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_loss_func = compute_loss_func
        self.loss_history = []
    def get_training_loss(self):
        return self.loss_history
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if is_datasets_available():
            import datasets
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "shuffle": self.args.shuffle,
        }

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        if self.compute_loss_func is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if labels is not None:
            loss = self.compute_loss_func(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss
    def compute_loss_unlabel(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        outputs = model(**inputs)
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.args.task_name == "stsb":  # 回归任务
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        else:  # 分类任务
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(outputs.logits.view(-1, self.model.config.num_labels), labels.view(-1))
            logits = outputs.logits
            logsoftmax_func = nn.LogSoftmax(dim=1)
            logsoftmax_logits = logsoftmax_func(logits)
            nllloss_func = nn.NLLLoss()
            label_loss = nllloss_func(logsoftmax_logits, labels)
            unlabel_loss = nllloss_func(logsoftmax_logits, 1 - labels)
            loss = label_loss + 1.0 / (self.args.reg * unlabel_loss)
        return (loss, outputs) if return_outputs else loss
    def training_step(self, model, inputs,num_items_in_batch=None) -> torch.Tensor:
        if self.args.state == "ft":
            model.train()
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            self.accelerator.backward(loss)
            if self.args.remain_loss:
                self.loss_history.append((loss.detach() / self.args.gradient_accumulation_steps).item())
            return loss.detach() / self.args.gradient_accumulation_steps
        elif self.args.state == "unlabel":
            model.train()
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss_unlabel(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            self.accelerator.backward(loss)
            if self.args.remain_loss:
                self.loss_history.append((loss.detach() / self.args.gradient_accumulation_steps).item())
            return loss.detach() / self.args.gradient_accumulation_steps
        elif self.args.state == "pm":
            model.eval()
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            return loss.detach() / self.args.gradient_accumulation_steps
        else:
            raise ValueError("state must be 'ft', 'unlabel' or 'pm'")
    def traing_loop(self,args=None):
        # !python roberta/train.py --state pm --dataset mrpc --seed 42 --reg 5e-7 --weight_decay 0.001 --epoch 1
        loss_before = []
        loss_after = []
        self._train_batch_size = args.per_device_train_batch_size
        train_dataloader = self.get_train_dataloader()
        len_dataloader = len(train_dataloader)
        epochs_trained = 0
        num_train_epochs = args.num_train_epochs
        num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

        if args.max_steps > 0:
            max_steps = args.max_steps
        else:
            max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
        delay_optimizer_creation =  self.is_fsdp_xla_enabled or self.is_fsdp_enabled
        # if self._created_lr_scheduler:
        #     self.lr_scheduler = None
        #     self._created_lr_scheduler = False
        # if not delay_optimizer_creation:
        #     self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        model = self._wrap_model(self.model_wrapped)
        use_accelerator_prepare = True if model is self.model else False
        # if use_accelerator_prepare:
        #     self.model.train()
        #     if hasattr(self.lr_scheduler, "step"):
        #         if self.use_apex:
        #             model = self.accelerator.prepare(self.model)
        #         else:
        #             model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        #     else:
        #         # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
        #         model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
        #             self.model, self.optimizer, self.lr_scheduler
        #         )
        # elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
        #     # In this case we are in DDP + LOMO, which should be supported
        #     self.optimizer = self.accelerator.prepare(self.optimizer)
        if model is not self.model:
            self.model_wrapped = model
        model.zero_grad()
        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            steps_in_epoch = len_dataloader
            epoch_iterator = iter(train_dataloader)
            num_examples = self.num_examples(train_dataloader)
            remainder = num_examples % args.gradient_accumulation_steps
            num_items_in_batch = None
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            #压缩前
            update_step = -1
            step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
            training_before = tqdm(total=total_updates, desc="training_before")
            for _ in range(total_updates):
                training_before.update(1)
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches)
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    total_batched_samples += 1
                    is_last_step_and_steps_less_than_grad_acc = (
                            steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                    )
                    do_sync_step = is_last_step_and_steps_less_than_grad_acc or (
                            total_batched_samples % args.gradient_accumulation_steps == 0
                    )
                    # if not do_sync_step:
                    #     self.accelerator.gradient_state._set_sync_gradients(False)
                    # else:
                    #     self.accelerator.gradient_state._set_sync_gradients(True)
                    tr_loss_step = self.training_step(model, inputs)
                    if do_sync_step:
                        loss_before.append(tr_loss_step.item())
                        del tr_loss_step
                        # self.accelerator.gradient_state._set_sync_gradients(True)
                        # self.optimizer.step()
                        # self.lr_scheduler.step()
                        # model.zero_grad()
            training_before.close()
            #压缩
            compress_ratio = args.reg
            with torch.no_grad():
                for n, p in model.named_modules():
                    if isinstance(p, (nn.Linear,Conv1D)):
                        p.weight.data = p.weight.data*(1-compress_ratio)
            # 压缩后
            epoch_iterator = iter(train_dataloader)
            update_step = -1
            step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
            training_after = tqdm(total=total_updates, desc="training_after")
            for _ in range(total_updates):
                training_after.update(1)
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches)
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    total_batched_samples += 1
                    is_last_step_and_steps_less_than_grad_acc = (
                            steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                    )
                    do_sync_step = is_last_step_and_steps_less_than_grad_acc or (
                            total_batched_samples % args.gradient_accumulation_steps == 0
                    )
                    if not do_sync_step:
                        self.accelerator.gradient_state._set_sync_gradients(False)
                    else:
                        self.accelerator.gradient_state._set_sync_gradients(True)
                    tr_loss_step = self.training_step(model, inputs)
                    if do_sync_step:
                        loss_after.append(tr_loss_step.item())
                        del tr_loss_step
            training_after.close()
        loss_gap = [(a - b) for a, b in zip(loss_after, loss_before)]
        loss_gap_file = f"{args.output_dir}/loss_gap_{args.task_name}_{args.weight_decay}_{args.reg}_{args.seed}.csv"
        df = pd.DataFrame(loss_gap)
        df.to_csv(loss_gap_file, index=False)
    def train(
        self,
        resume_from_checkpoint = None,
        trial = None,
        ignore_keys_for_eval = None,
        **kwargs,
    ):
        """
        Main training entry point.
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        args = self.args

        self.is_in_train = True

        # Attach NEFTune hooks if necessary
        if self.neftune_noise_alpha is not None:
            self.model = self._activate_neftune(self.model)

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train and not self.is_model_parallel:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model
        if args.state == 'pm':
            inner_training_loop = self.traing_loop
            return inner_training_loop(args=args)
        else:
            inner_training_loop = find_executable_batch_size(self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size)
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )


