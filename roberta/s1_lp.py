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
from torch.utils.data.dataloader import DataLoader
import time
from transformers.pytorch_utils import  Conv1D
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
    data_p = GLUEPruner(dataset=trainset, ratio=config.target_ratio,pruneFlag=config.pruneFlag)


    data_p.prune()
    sampler = data_p.get_sampler()
    train_epoch_iterator = get_pruned_dataloader(config, trainset, sampler) if config.target_ratio != 0 else train_dataloader

    #微调几个epoch

    loss_before = {}
    loss_after = {}
    loss_g_before = {}
    loss_g_after = {}
    logits_before = {}
    logits_after = {}
    labels_before = {}
    labels_after = {}
    iterator = iter(train_epoch_iterator)
    trange = range(len(train_epoch_iterator))
    before = tqdm(total=len(train_epoch_iterator), desc="before")
    for step in trange:
        before.update(1)
        inputs = prepare_inputs(next(iterator), device)
        model.eval()
        step_idx = inputs["idx"]

        loss = compute_loss(model, inputs)

        for i in range(len(step_idx)):
            loss_g_before[step_idx[i].item()] = loss.item()
            labels_before[step_idx[i].item()] = inputs['labels'][i]

        logits,outputs = compute_logits_pooler_output(model, inputs)
        # print(logits.data)
        logits = logits.data.reshape(-1, 2)

        outputs = outputs.reshape(-1, 768)
        assert len(step_idx) == outputs.shape[0], "The length of step_idx must match the number of rows in outputs"
        outputs = outputs.data.cpu()
        for i in range(len(step_idx)):
            loss_before[step_idx[i].item()] = outputs[i]
            logits_before[step_idx[i].item()] = logits[i]
        del outputs,logits


    before.close()
    
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear,Conv1D)):
                r = 1 - compress
                module.weight.data = r * module.weight.data

    iterator = iter(train_epoch_iterator)
    trange = range(len(train_epoch_iterator))
    after = tqdm(total=len(train_epoch_iterator), desc="after")
    for step in trange:
        after.update(1)
        inputs = prepare_inputs(next(iterator), device)
        model.eval()
        step_idx = inputs["idx"]
        loss = compute_loss(model, inputs)

        for i in range(len(step_idx)):
            loss_g_after[step_idx[i].item()] = loss.item()
            labels_after[step_idx[i].item()] = inputs['labels'][i]

        logits,outputs = compute_logits_pooler_output(model, inputs)
        logits = logits.data.reshape(-1, 2)
        outputs = outputs.data.reshape(-1, 768)
        assert len(step_idx) == outputs.shape[0], "The length of step_idx must match the number of rows in outputs"
        outputs = outputs.data.cpu()
        for i in range(len(step_idx)):
            loss_after[step_idx[i].item()] = outputs[i]
            logits_after[step_idx[i].item()] = logits[i]
        del outputs,logits

    after.close()
    loss_gap = {key: torch.sum(torch.abs(loss_after[key] - loss_before[key])).item() for key in loss_after if key in loss_before}
    loss_g_gap = {key: loss_g_after[key] - loss_g_before[key] for key in loss_after if key in loss_before}
    logits_gap = {}
    for key in logits_after.keys():
        after_values = logits_after[key]
        before_values = logits_before[key]
        diff1 = after_values[0] - before_values[0]
        diff2 = after_values[1] - before_values[1]
        logits_gap[key] = [diff1.item(),diff2.item()]


    score = loss_gap
    modelname = config.model
    if 't5' in config.model:
        modelname = "t5"
    keys = sorted(loss_gap.keys())
    result_array = [[loss_gap[key], loss_g_gap[key],logits_gap[key][0],logits_gap[key][1],logits_before[key][0].item(),logits_before[key][1].item(),logits_after[key][0].item(),logits_after[key][1].item(),labels_before[key].item(),labels_after[key].item()] for key in keys]

    scores_file = f"s1_{modelname}_{config['dataset']}_{config['seed']}_{config['reg']}_{config['batchsize']}.csv"
    df = pd.DataFrame(result_array, columns=["s1", "loss_gap","logit_gap0","logit_gap1","logit_before0","logit_before1","logit_after0","logit_after1","labels_before","labels_after"])
    df.to_csv(scores_file, index=False)

    end_time = time.time()
    total_time = end_time - start_time
    s = f'Total training time: {total_time}'
    log.info(s)


if __name__ == "__main__":
    main()
    #每个样本的损失颗粒和输出颗粒的一范数
    # python roberta/s1_lp.py --state ft --dataset mrpc --seed 3404 --reg 5e-8 --weight_decay 0.0 --epoch 1 --model bert-base-uncased --batchsize 1 --shuffle False
