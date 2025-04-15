# from transformers import pipeline,AutoModelForSequenceClassification,AutoModelForSeq2SeqLM
import numpy as np
a=[1,2,3,4,5]
b=a[1:3]
c=a[-5:-1]
# print(a,b,c)
a=2
b=10**a
# print(b)

import torch

# 创建一个 3 维张量（例如 2x3x4 的张量）
tensor = torch.rand(2, 3, 4)  # 随机生成一个张量

# 计算 Frobenius 范数
frobenius_norm = torch.norm(tensor, p='fro')

print("张量的 Frobenius 范数为:", frobenius_norm)
tensor = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
sum = torch.norm(tensor, p=1).item()
print(sum)