import torch
import numpy as np
import heapq
from datasets import Dataset
import operator
# from evaluate import load
tasks = ['mrpc','sst2','rte','cola','stsb','qqp','mnli','qnli']
# for task in tasks:
#     metric = load("glue", task)
#     print(f'{task}:{metric}')
# # scores = np.arange(100)
# print(scores)
# largest = heapq.nlargest(5, enumerate(scores), key=lambda x: x[1])
# largest_sorted = sorted(largest, key=lambda x: x[1], reverse=True)
# indices = [index for index, score in largest_sorted]
# data = {
#     'idx': np.arange(100),  # idx 从 0 到 99
#     'input_ids': np.arange(100),  # 假设每个样本有10个输入ID
# }
#
# # 创建Dataset对象
# dataset = Dataset.from_dict(data)
# print(dataset[indices])
#
# from transformers.data.data_collator import DataCollatorWithPadding
# from torch.utils.data.dataloader import DataLoader
# from transformers import AutoTokenizer
# model_checkpoint="bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# data_collator = DataCollatorWithPadding(tokenizer)
#
# inputdata_dataloader = DataLoader(
#     dataset,
#     shuffle=False,
#     batch_size=5,
#     collate_fn=data_collator,
#     # drop_last=dataloader_drop_last,
#     num_workers=1,
#     pin_memory=True
# )
# iterator = iter(inputdata_dataloader)
# print(next(iterator))
# inputs = next(iterator)
# get_score = operator.itemgetter(*inputs['idx'].tolist())
# step_score = torch.tensor(get_score(scores))
# print(step_score,inputs['idx'])

import torch

# 创建一个张量示例
x = torch.tensor([[1.0, 2.0, 3.0],[4,5,6]])

# 计算二范数
norm_2 = torch.norm(x, dim=1)
norm_3 = torch.norm(x)
print(norm_2)
print(norm_3)
