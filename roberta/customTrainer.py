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
def prepare_inputs(inputs, device):
    """
    Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs

class GlueTrainingArguments(TrainingArguments):
    def __init__(self,dynamic:bool =False,model_name:str ='', task_name: str = '', state: str = '', reg: int = 1, shuffle:bool =True, remain_loss:bool = False,**kwargs):
        super().__init__(**kwargs)
        self.task_name = task_name
        self.state = state
        self.reg = reg
        self.shuffle = shuffle
        self.remain_loss = remain_loss
        self.dynamic = dynamic
        self.model_name = model_name

class GlueTrainer(Trainer):
    def __init__(self, compute_loss_func = None,pruner = None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_loss_func = compute_loss_func
        self.pruner = pruner
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
        # if self.args.dynamic:
        #     from model_loader import get_model_and_tokenizer
        #     import operator
        #     model_checkpoint = self.args.model_name
        #     task = self.args.task_name
        #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #     compress = self.args.reg
        #     prune_model, _ = get_model_and_tokenizer(model_checkpoint, task, device)
        #     train_epoch_iterator = DataLoader(
        #         self.train_dataset,
        #         shuffle=False,
        #         batch_size=1,
        #         collate_fn=self.data_collator,
        #         num_workers=self.args.dataloader_num_workers,
        #         pin_memory=self.args.dataloader_pin_memory
        #     )
        #     loss_before = {}
        #     loss_after = {}
        #     iterator = iter(train_epoch_iterator)
        #     trange = range(len(train_epoch_iterator))
        #
        #     before = tqdm(total=len(train_epoch_iterator), desc=f"lp before")
        #     for step in trange:
        #         before.update(1)
        #         inputs = prepare_inputs(next(iterator), device)
        #         prune_model.eval()
        #         step_idx = inputs["idx"]
        #         loss = self.compute_loss(prune_model, inputs)
        #         for i in range(len(step_idx)):
        #             loss_before[step_idx[i].item()] = loss.data
        #     before.close()
        #     with torch.no_grad():
        #         for name, module in prune_model.named_modules():
        #             if isinstance(module, torch.nn.Linear):
        #                 r = 1 - compress
        #                 module.weight.data = r * module.weight.data
        #     iterator = iter(train_epoch_iterator)
        #     trange = range(len(train_epoch_iterator))
        #     after = tqdm(total=len(train_epoch_iterator), desc=f"lp after")
        #     for step in trange:
        #         after.update(1)
        #         inputs = prepare_inputs(next(iterator), device)
        #         prune_model.eval()
        #         step_idx = inputs["idx"]
        #         loss = self.compute_loss(prune_model, inputs)
        #         for i in range(len(step_idx)):
        #             loss_after[step_idx[i].item()] = loss.data
        #     after.close()
        #     loss_gap = {key: torch.abs(loss_after[key] - loss_before[key]).item() for key in loss_after if key in loss_before}
        #     iterator = iter(train_epoch_iterator)
        #     trange = range(len(train_epoch_iterator))
        #     print(len(train_epoch_iterator))
        #     for step in trange:
        #         inputs = prepare_inputs(next(iterator), device)
        #         step_size = len(inputs['idx'])
        #         step_score = torch.randint(1, 1001, (step_size,))
        #         get_score = operator.itemgetter(*inputs['idx'].tolist())
        #         step_score = torch.tensor(get_score(loss_gap))
        #         self.pruner.update(step_score, inputs['idx'])
        #     print(f'修剪前：{len(self.pruner.cur_index)}')
        #     self.pruner.prune()
        #     print(f'修剪后：{len(self.pruner.cur_index)}')
        #     train_dataset = self.pruner.get_pruned_train_dataset()
        #     del prune_model
        if self.args.dynamic:
            from model_loader import get_model_and_tokenizer
            import operator
            model_checkpoint = self.args.model_name
            task = self.args.task_name
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            compress = self.args.reg
            prune_model, _ = get_model_and_tokenizer(model_checkpoint, task, device)
            train_epoch_iterator = DataLoader(
                self.train_dataset,
                shuffle=False,
                batch_size=1,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory
            )
            loss_before = {}
            loss_after = {}
            iterator = iter(train_epoch_iterator)
            trange = range(len(train_epoch_iterator))

            before = tqdm(total=len(train_epoch_iterator), desc=f"lp before")
            for step in trange:
                before.update(1)
                inputs = prepare_inputs(next(iterator), device)
                prune_model.eval()
                step_idx = inputs["idx"]
                loss = self.compute_loss(prune_model, inputs)
                for i in range(len(step_idx)):
                    loss_before[step_idx[i].item()] = loss.data
            before.close()
            with torch.no_grad():
                for name, module in prune_model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        r = 1 - compress
                        module.weight.data = r * module.weight.data
            iterator = iter(train_epoch_iterator)
            trange = range(len(train_epoch_iterator))
            after = tqdm(total=len(train_epoch_iterator), desc=f"lp after")
            for step in trange:
                after.update(1)
                inputs = prepare_inputs(next(iterator), device)
                prune_model.eval()
                step_idx = inputs["idx"]
                loss = self.compute_loss(prune_model, inputs)
                for i in range(len(step_idx)):
                    loss_after[step_idx[i].item()] = loss.data
            after.close()
            loss_gap = {key: torch.abs(loss_after[key] - loss_before[key]).item() for key in loss_after if key in loss_before}
            iterator = iter(train_epoch_iterator)
            trange = range(len(train_epoch_iterator))
            print(len(train_epoch_iterator))
            index = np.array([key for key, value in loss_gap.items() if value != 0])
            print(f'修剪前：{len(self.pruner.cur_index)}')
            self.pruner.lp_prune(index)
            print(f'修剪后：{len(self.pruner.cur_index)}')
            train_dataset = self.pruner.get_pruned_train_dataset()
            del prune_model
        else:
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
        if "idx" in inputs:
            idx = inputs.pop("idx")
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
    # def inner_training_loop(
    #     self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    # ):
    #     self.accelerator.free_memory()
    #     self._train_batch_size = batch_size
    #     if self.args.auto_find_batch_size:
    #         if self.state.train_batch_size != self._train_batch_size:
    #             from accelerate.utils import release_memory
    #
    #             (self.model_wrapped,) = release_memory(self.model_wrapped)
    #             self.model_wrapped = self.model
    #
    #             # Check for DeepSpeed *after* the intial pass and modify the config
    #             if self.is_deepspeed_enabled:
    #                 # Temporarily unset `self.args.train_batch_size`
    #                 original_bs = self.args.per_device_train_batch_size
    #                 self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
    #                 self.propagate_args_to_deepspeed(True)
    #                 self.args.per_device_train_batch_size = original_bs
    #         self.state.train_batch_size = self._train_batch_size
    #     logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
    #     # Data loader and number of training steps
    #     train_dataloader = self.get_train_dataloader()
    #     if self.is_fsdp_xla_v2_enabled:
    #         train_dataloader = tpu_spmd_dataloader(train_dataloader)
    #
    #     # Setting up training control variables:
    #     # number of training epochs: num_train_epochs
    #     # number of training steps per epoch: num_update_steps_per_epoch
    #     # total number of training steps to execute: max_steps
    #     total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
    #
    #     len_dataloader = None
    #     num_train_tokens = None
    #     if has_length(train_dataloader):
    #         len_dataloader = len(train_dataloader)
    #         num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
    #         num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    #         num_examples = self.num_examples(train_dataloader)
    #         if args.max_steps > 0:
    #             max_steps = args.max_steps
    #             num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
    #                 args.max_steps % num_update_steps_per_epoch > 0
    #             )
    #             # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
    #             # the best we can do.
    #             num_train_samples = args.max_steps * total_train_batch_size
    #             if args.include_tokens_per_second:
    #                 num_train_tokens = (
    #                     self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
    #                 )
    #         else:
    #             max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    #             num_train_epochs = math.ceil(args.num_train_epochs)
    #             num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
    #             if args.include_tokens_per_second:
    #                 num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
    #     elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
    #         max_steps = args.max_steps
    #         # Setting a very large number of epochs so we go as many times as necessary over the iterator.
    #         num_train_epochs = sys.maxsize
    #         num_update_steps_per_epoch = max_steps
    #         num_examples = total_train_batch_size * args.max_steps
    #         num_train_samples = args.max_steps * total_train_batch_size
    #         if args.include_tokens_per_second:
    #             num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
    #     else:
    #         raise ValueError(
    #             "args.max_steps must be set to a positive value if dataloader does not have a length, was"
    #             f" {args.max_steps}"
    #         )
    #
    #     if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
    #         if self.args.n_gpu > 1:
    #             # nn.DataParallel(model) replicates the model, creating new variables and module
    #             # references registered here no longer work on other gpus, breaking the module
    #             raise ValueError(
    #                 "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
    #                 " (torchrun or torch.distributed.launch (deprecated))."
    #             )
    #         else:
    #             debug_overflow = DebugUnderflowOverflow(self.model)  # noqa
    #
    #     delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled
    #
    #     # We need to reset the scheduler, as its parameters may be different on subsequent calls
    #     if self._created_lr_scheduler:
    #         self.lr_scheduler = None
    #         self._created_lr_scheduler = False
    #
    #     if self.is_deepspeed_enabled:
    #         self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)
    #
    #     if not delay_optimizer_creation:
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)
    #
    #     self.state = TrainerState(
    #         stateful_callbacks=[
    #             cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
    #         ]
    #     )
    #     self.state.is_hyper_param_search = trial is not None
    #     self.state.train_batch_size = self._train_batch_size
    #
    #     # Compute absolute values for logging, eval, and save if given as ratio
    #     if args.logging_steps is not None:
    #         if args.logging_steps < 1:
    #             self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
    #         else:
    #             self.state.logging_steps = args.logging_steps
    #     if args.eval_steps is not None:
    #         if args.eval_steps < 1:
    #             self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
    #         else:
    #             self.state.eval_steps = args.eval_steps
    #     if args.save_steps is not None:
    #         if args.save_steps < 1:
    #             self.state.save_steps = math.ceil(max_steps * args.save_steps)
    #         else:
    #             self.state.save_steps = args.save_steps
    #
    #     # Activate gradient checkpointing if needed
    #     if args.gradient_checkpointing:
    #         self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)
    #
    #     model = self._wrap_model(self.model_wrapped)
    #
    #     # as the model is wrapped, don't use `accelerator.prepare`
    #     # this is for unhandled cases such as
    #     # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
    #     use_accelerator_prepare = True if model is self.model else False
    #
    #     if delay_optimizer_creation:
    #         if use_accelerator_prepare:
    #             self._fsdp_qlora_plugin_updates()
    #             self.model = self.accelerator.prepare(self.model)
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)
    #
    #     # prepare using `accelerator` prepare
    #     if use_accelerator_prepare:
    #         self.model.train()
    #         if hasattr(self.lr_scheduler, "step"):
    #             if self.use_apex:
    #                 model = self.accelerator.prepare(self.model)
    #             else:
    #                 model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
    #         else:
    #             # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
    #             model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
    #                 self.model, self.optimizer, self.lr_scheduler
    #             )
    #     elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
    #         # In this case we are in DDP + LOMO, which should be supported
    #         self.optimizer = self.accelerator.prepare(self.optimizer)
    #
    #     if self.is_fsdp_enabled:
    #         self.model = self.model_wrapped = model
    #
    #     # for the rest of this function `model` is the outside model, whether it was wrapped or not
    #     if model is not self.model:
    #         self.model_wrapped = model
    #
    #     # backward compatibility
    #     if self.is_deepspeed_enabled:
    #         self.deepspeed = self.model_wrapped
    #
    #     # ckpt loading
    #     if resume_from_checkpoint is not None:
    #         if self.is_deepspeed_enabled:
    #             deepspeed_load_checkpoint(
    #                 self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
    #             )
    #         elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
    #             self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)
    #
    #     # Check if saved optimizer or scheduler states exist
    #     self._load_optimizer_and_scheduler(resume_from_checkpoint)
    #
    #     # important: at this point:
    #     # self.model         is the Transformers Model
    #     # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
    #     # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.
    #
    #     # Train!
    #     logger.info("***** Running training *****")
    #     logger.info(f"  Num examples = {num_examples:,}")
    #     logger.info(f"  Num Epochs = {num_train_epochs:,}")
    #     logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
    #     if self.args.per_device_train_batch_size != self._train_batch_size:
    #         logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
    #     logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
    #     logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    #     logger.info(f"  Total optimization steps = {max_steps:,}")
    #     logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")
    #
    #     self.state.epoch = 0
    #     start_time = time.time()
    #     epochs_trained = 0
    #     steps_trained_in_current_epoch = 0
    #     steps_trained_progress_bar = None
    #
    #     # Check if continuing training from a checkpoint
    #     if resume_from_checkpoint is not None and os.path.isfile(
    #         os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
    #     ):
    #         self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
    #         self.compare_trainer_and_checkpoint_args(self.args, self.state)
    #         self._load_callback_state()
    #         epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
    #         if not args.ignore_data_skip:
    #             steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
    #             steps_trained_in_current_epoch *= args.gradient_accumulation_steps
    #         else:
    #             steps_trained_in_current_epoch = 0
    #
    #         logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #         logger.info(f"  Continuing training from epoch {epochs_trained}")
    #         logger.info(f"  Continuing training from global step {self.state.global_step}")
    #         if not args.ignore_data_skip:
    #             logger.info(
    #                 f"  Will skip the first {epochs_trained} epochs then the first"
    #                 f" {steps_trained_in_current_epoch} batches in the first epoch."
    #             )
    #
    #     # Update the references
    #     self.callback_handler.model = self.model
    #     self.callback_handler.optimizer = self.optimizer
    #     self.callback_handler.lr_scheduler = self.lr_scheduler
    #     self.callback_handler.train_dataloader = train_dataloader
    #     if self.hp_name is not None and self._trial is not None:
    #         # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
    #         # parameter to Train when using DDP.
    #         self.state.trial_name = self.hp_name(self._trial)
    #     if trial is not None:
    #         assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
    #         self.state.trial_params = hp_params(assignments)
    #     else:
    #         self.state.trial_params = None
    #     # This should be the same if the state has been saved but in case the training arguments changed, it's safer
    #     # to set this after the load.
    #     self.state.max_steps = max_steps
    #     self.state.num_train_epochs = num_train_epochs
    #     self.state.is_local_process_zero = self.is_local_process_zero()
    #     self.state.is_world_process_zero = self.is_world_process_zero()
    #
    #     # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    #     tr_loss = torch.tensor(0.0).to(args.device)
    #     # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    #     self._total_loss_scalar = 0.0
    #     self._globalstep_last_logged = self.state.global_step
    #     model.zero_grad()
    #     grad_norm: Optional[float] = None
    #     self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
    #
    #     if args.eval_on_start:
    #         self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)
    #
    #     total_batched_samples = 0
    #     for epoch in range(epochs_trained, num_train_epochs):
    #         epoch_dataloader = train_dataloader
    #         if hasattr(epoch_dataloader, "set_epoch"):
    #             epoch_dataloader.set_epoch(epoch)
    #
    #         # Reset the past mems state at the beginning of each epoch if necessary.
    #         if args.past_index >= 0:
    #             self._past = None
    #
    #         steps_in_epoch = (
    #             len(epoch_dataloader)
    #             if len_dataloader is not None
    #             else args.max_steps * args.gradient_accumulation_steps
    #         )
    #         self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
    #
    #         if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
    #             self._load_rng_state(resume_from_checkpoint)
    #
    #         rng_to_sync = False
    #         steps_skipped = 0
    #         if steps_trained_in_current_epoch > 0:
    #             epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
    #             steps_skipped = steps_trained_in_current_epoch
    #             steps_trained_in_current_epoch = 0
    #             rng_to_sync = True
    #
    #         step = -1
    #         epoch_iterator = iter(epoch_dataloader)
    #         # We chunkify the epoch iterator into gradient accumulation steps `n` batches
    #         remainder = num_examples % args.gradient_accumulation_steps
    #         num_items_in_batch = None
    #         if remainder == 0:
    #             remainder = args.gradient_accumulation_steps
    #         update_step = -1
    #         total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
    #         for _ in range(total_updates):
    #             update_step += 1
    #             num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
    #             batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches)
    #             for i, inputs in enumerate(batch_samples):
    #                 step += 1
    #                 total_batched_samples += 1
    #                 is_last_step_and_steps_less_than_grad_acc = (
    #                     steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
    #                 )
    #                 do_sync_step = is_last_step_and_steps_less_than_grad_acc or (
    #                     total_batched_samples % args.gradient_accumulation_steps == 0
    #                 )
    #                 # Since we perform prefetching, we need to manually set sync_gradients
    #                 if not do_sync_step:
    #                     self.accelerator.gradient_state._set_sync_gradients(False)
    #                 else:
    #                     self.accelerator.gradient_state._set_sync_gradients(True)
    #
    #                 if self.args.include_num_input_tokens_seen:
    #                     main_input_name = getattr(self.model, "main_input_name", "input_ids")
    #                     if main_input_name not in inputs:
    #                         logger.warning(
    #                             "Tried to track the number of tokens seen, however the current model is "
    #                             "not configured properly to know what item is the input. To fix this, add "
    #                             "a `main_input_name` attribute to the model class you are using."
    #                         )
    #                     else:
    #                         input_tokens = inputs[main_input_name].numel()
    #                         input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
    #                         self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).cpu().item()
    #                 if rng_to_sync:
    #                     self._load_rng_state(resume_from_checkpoint)
    #                     rng_to_sync = False
    #
    #                 # Skip past any already trained steps if resuming training
    #                 if steps_trained_in_current_epoch > 0:
    #                     steps_trained_in_current_epoch -= 1
    #                     if steps_trained_progress_bar is not None:
    #                         steps_trained_progress_bar.update(1)
    #                     if steps_trained_in_current_epoch == 0:
    #                         self._load_rng_state(resume_from_checkpoint)
    #                     continue
    #                 elif steps_trained_progress_bar is not None:
    #                     steps_trained_progress_bar.close()
    #                     steps_trained_progress_bar = None
    #
    #                 if step % args.gradient_accumulation_steps == 0:
    #                     self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
    #
    #                 # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
    #                 context = (
    #                     functools.partial(self.accelerator.no_sync, model=model)
    #                     if i != len(batch_samples) - 1
    #                     else contextlib.nullcontext
    #                 )
    #                 with context():
    #                     tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
    #
    #                 if (
    #                     args.logging_nan_inf_filter
    #                     and not is_torch_xla_available()
    #                     and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
    #                 ):
    #                     # if loss is nan or inf simply add the average of previous logged losses
    #                     tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
    #                 else:
    #                     if tr_loss.device != tr_loss_step.device:
    #                         raise ValueError(
    #                             f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
    #                         )
    #                     tr_loss = tr_loss + tr_loss_step
    #
    #                 self.current_flos += float(self.floating_point_ops(inputs))
    #
    #                 if do_sync_step:
    #                     # Since we perform prefetching, we need to manually set sync_gradients to True
    #                     self.accelerator.gradient_state._set_sync_gradients(True)
    #
    #                     # Gradient clipping
    #                     if args.max_grad_norm is not None and args.max_grad_norm > 0:
    #                         # deepspeed does its own clipping
    #
    #                         if is_sagemaker_mp_enabled() and args.fp16:
    #                             _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
    #                         elif self.use_apex:
    #                             # Revert to normal clipping otherwise, handling Apex or full precision
    #                             _grad_norm = nn.utils.clip_grad_norm_(
    #                                 amp.master_params(self.optimizer),
    #                                 args.max_grad_norm,
    #                             )
    #                         else:
    #                             _grad_norm = self.accelerator.clip_grad_norm_(
    #                                 model.parameters(),
    #                                 args.max_grad_norm,
    #                             )
    #
    #                         if (
    #                             is_accelerate_available()
    #                             and self.accelerator.distributed_type == DistributedType.DEEPSPEED
    #                         ):
    #                             grad_norm = model.get_global_grad_norm()
    #                             # In some cases the grad norm may not return a float
    #                             if hasattr(grad_norm, "item"):
    #                                 grad_norm = grad_norm.item()
    #                         else:
    #                             grad_norm = _grad_norm
    #
    #                     self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)
    #
    #                     self.optimizer.step()
    #
    #                     self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)
    #
    #                     optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
    #                     if optimizer_was_run:
    #                         # Delay optimizer scheduling until metrics are generated
    #                         if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
    #                             self.lr_scheduler.step()
    #
    #                     model.zero_grad()
    #                     self.state.global_step += 1
    #                     self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
    #                     self.control = self.callback_handler.on_step_end(args, self.state, self.control)
    #                     self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
    #                 else:
    #                     self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
    #
    #                 # PyTorch/XLA relies on the data loader to insert the mark_step for
    #                 # each step. Since we are breaking the loop early, we need to manually
    #                 # insert the mark_step here.
    #                 if self.control.should_epoch_stop or self.control.should_training_stop:
    #                     if is_torch_xla_available():
    #                         xm.mark_step()
    #                     break
    #             # We also need to break out of the nested loop
    #             if self.control.should_epoch_stop or self.control.should_training_stop:
    #                 if is_torch_xla_available():
    #                     xm.mark_step()
    #                 break
    #         if step < 0:
    #             logger.warning(
    #                 "There seems not to be a single sample in your epoch_iterator, stopping training at step"
    #                 f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
    #                 f" num_steps ({max_steps}) higher than the number of available samples."
    #             )
    #             self.control.should_training_stop = True
    #
    #         self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
    #         self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
    #
    #         if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
    #             if is_torch_xla_available():
    #                 # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
    #                 xm.master_print(met.metrics_report())
    #             else:
    #                 logger.warning(
    #                     "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
    #                     "configured. Check your training configuration if this is unexpected."
    #                 )
    #         if self.control.should_training_stop:
    #             break
    #
    #     if args.past_index and hasattr(self, "_past"):
    #         # Clean the state at the end of training
    #         delattr(self, "_past")
    #
    #     logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    #     if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
    #         # Wait for everyone to get here so we are sure the model has been saved by process 0.
    #         if is_torch_xla_available():
    #             xm.rendezvous("load_best_model_at_end")
    #         elif args.parallel_mode == ParallelMode.DISTRIBUTED:
    #             dist.barrier()
    #         elif is_sagemaker_mp_enabled():
    #             smp.barrier()
    #
    #         self._load_best_model()
    #
    #     # add remaining tr_loss
    #     self._total_loss_scalar += tr_loss.item()
    #     effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
    #     train_loss = self._total_loss_scalar / effective_global_step
    #
    #     metrics = speed_metrics(
    #         "train",
    #         start_time,
    #         num_samples=num_train_samples,
    #         num_steps=self.state.max_steps,
    #         num_tokens=num_train_tokens,
    #     )
    #     self.store_flos()
    #     metrics["total_flos"] = self.state.total_flos
    #     metrics["train_loss"] = train_loss
    #
    #     self.is_in_train = False
    #
    #     self._memory_tracker.stop_and_update_metrics(metrics)
    #
    #     self.log(metrics)
    #
    #     run_dir = self._get_output_dir(trial)
    #     checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)
    #
    #     # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
    #     if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
    #         for checkpoint in checkpoints_sorted:
    #             if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
    #                 logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
    #                 shutil.rmtree(checkpoint, ignore_errors=True)
    #
    #     self.control = self.callback_handler.on_train_end(args, self.state, self.control)
    #
    #     # Wait for the checkpoint to be uploaded.
    #     self._finish_current_push()
    #
    #     # After training we make sure to retrieve back the original forward pass method
    #     # for the embedding layer by removing the forward post hook.
    #     if self.neftune_noise_alpha is not None:
    #         self._deactivate_neftune(self.model)
    #
    #     return TrainOutput(self.state.global_step, train_loss, metrics)
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


