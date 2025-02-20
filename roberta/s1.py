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
            if isinstance(module, (torch.nn.Linear,Conv1D)):
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
    if config.pruneFlag=="random":
        common_keys = [key for key in loss_after if key in loss_before]
        loss_gap = {key: index for index, key in enumerate(common_keys)}
    else:
        # loss_gap = {key: torch.max(torch.abs(loss_after[key] - loss_before[key])).item() for key in loss_after.keys() if key in loss_before.keys()}
        loss_gap = {key: torch.sum(torch.abs(loss_after[key] - loss_before[key])).item() for key in loss_after if key in loss_before}
        #欧氏距离
        # loss_gap = {key: torch.sqrt(torch.sum(torch.square(loss_after[key] - loss_before[key]))).item() for key in loss_after if key in loss_before}
        # loss_gap = {key: torch.var(loss_after[key] - loss_before[key]).item() for key in loss_after if key in loss_before}
    score = loss_gap
    # iterator = iter(train_epoch_iterator)
    # loss_before = []
    # loss_after = []
    # model2, _ = get_model_and_tokenizer(model_checkpoint, task, device)
    # model3, _ = get_model_and_tokenizer(model_checkpoint, task, device)
    # model3.load_state_dict(copy.deepcopy(model2.state_dict()))
    # model3.to(next(model2.parameters()).device)
    # inputs = prepare_inputs(next(iterator), device)
    # model.eval()
    # outputs = get_pooler_output(model, inputs)
    # outputs = outputs.data.cpu()
    # loss_before.append(outputs)
    # with torch.no_grad():
    #     for name, module in model.named_modules():
    #         if isinstance(module, (torch.nn.Linear,Conv1D)):
    #             r = 1 - compress
    #             module.weight.data = r * module.weight.data
    # model.eval()
    # outputs = get_pooler_output(model, inputs)
    # outputs = outputs.data.cpu()
    # loss_after.append(outputs)
    modelname = config.model
    if 't5' in config.model:
        modelname = "t5"
    # for i in range(len(loss_after)):
    pooler_gap = [v.item() for v in sum([torch.abs(loss_after[key] - loss_before[key]) for key in loss_after if key in loss_before])/len(loss_before)]
    pooler_gap_file = f"pooler_gap_{modelname}_{config.dataset}_{config.seed}_{config.reg}_{config.batchsize}.csv"
    df = pd.DataFrame(pooler_gap)
    df.to_csv(pooler_gap_file, index=False)
    scores = [score[key] for key in score]
    scores_file = f"scores_{modelname}_{config.dataset}_{config.seed}_{config.reg}_{config.batchsize}.csv"
    df = pd.DataFrame(scores)
    df.to_csv(scores_file, index=False)
    end_time = time.time()
    total_time = end_time - start_time
    s = f'Total training time: {total_time}'
    log.info(s)


if __name__ == "__main__":
    main()
    #保存s1，以及32个样本的768维颗粒
    # python roberta/s1.py --state ft --dataset mrpc --seed 3404 --reg 5e-8 --weight_decay 0.0 --epoch 1 --remain_loss 1 --model bert-base-uncased
