# -- coding: utf-8 --
from torch.utils.data.dataloader import DataLoader
import numpy as np
from transformers.data.data_collator import DataCollatorWithPadding
import torch
from transformers import BertTokenizer,AutoTokenizer
from torch.utils.data import SubsetRandomSampler
from datasets import Dataset

class GLUEPruner():
    def __init__(self, dataset,ratio = 0.5):
        self.dataset = dataset
        self.keep_ratio =1-ratio if ratio<=1 else (100-ratio)/100
        self.index=range(len(self.dataset))
        self.device=torch.device('cuda:0')
        self.scores = torch.zeros([len(self.dataset)]).cpu()
        self.cur_batch_index = None
        self.iteration=0
        self.cur_index=None
        self.num_pruned_samples=0

    def update(self, values,index):
        self.cur_batch_index=index
        s=torch.abs(values.to(dtype=self.scores.dtype).cpu())
        self.scores[self.cur_batch_index.cpu()]=s
        # self.scores[self.cur_batch_index.cpu()] = s



    def prune(self):
        if self.iteration==0:
            remain_indices=np.arange(len(self.dataset))
            np.random.shuffle(remain_indices)
            self.cur_index = remain_indices
            self.iteration += 1
            return
        def fraction_threshold(tensor, fraction):
            threshold, _ = torch.topk(tensor, int((fraction) * len(tensor)))
            return threshold[-1]

        def threshold_mask(tensor, threshold):
            assert isinstance(tensor, torch.Tensor)
            idx = tensor < threshold
            mask = torch.ones_like(tensor, device=torch.device('cuda:0'))
            mask[idx] = 0
            return mask
        print(self.keep_ratio)
        threshold = fraction_threshold(self.scores, self.keep_ratio)

        remain_mask = threshold_mask(self.scores, threshold).cpu().numpy().astype(bool)

        remain_indices = np.where(remain_mask)[0]
        print('remain_samples',len(remain_indices))
        np.random.shuffle(remain_indices)
        self.cur_index=remain_indices
        self.iteration+=1

    def pro_loss(self,loss):
        loss *= 1 / self.keep_ratio
        return loss

    def get_sampler(self):
        sampler = SubsetRandomSampler(self.cur_index)
        return sampler

    @property
    def save_ratio(self):
        return (self.num_pruned_samples)/(self.iteration*len(self.dataset))
    def get_pruned_train_dataset(self):
        traindataset = {k: [v[i] for i in self.cur_index] for k, v in self.dataset.items()}
        train_dataset = Dataset.from_dict(traindataset)
        return train_dataset
def get_pruned_dataloader(config,dataset,sampler):
    model_checkpoint = "bert-base-uncased"
    batch_size = config.batchsize
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
        # drop_last=True,
        num_workers=0,
        pin_memory=True, sampler=sampler
    )
    return train_dataloader


from transformers import  default_data_collator
def get_squad_dataloader(config,dataset,sampler):
    train_dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=config.batchsize,
    )
    return  train_dataloader





