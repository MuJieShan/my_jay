import torch
# import torchvision
import torch.nn as nn
import torch.optim as optim
# from transformers import LlamaModel,GPT2ForSequenceClassification,T5ForSequenceClassification
# from transformers.pytorch_utils import  Conv1D
# # # 从 torchvision 中获取 VGG16 模型
# # model = torchvision.models.vgg16(pretrained=False)
# #
# # # 定义损失函数和优化器
# # loss_func = nn.CrossEntropyLoss()
# # optimizer = optim.SGD(model.parameters(), lr=0.001,weight_decay=0.001)
# linear = nn.Linear(in_features=4,out_features=3)
# # print(linear.weight, linear.weight.data,linear.bias)
# linear.weight.data = linear.weight.data*3
# # print(linear.weight, linear.weight.data,linear.bias)
# LayerNorm = nn.LayerNorm(4)
# # print(LayerNorm.weight, LayerNorm.bias)
# Conv2d = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
# # print(Conv2d.weight.size())
# a=torch.zeros(3668,768,dtype=torch.float32)
dict_data = {0: 2, 2: 3, 1: 4}

# 对字典的 key 进行排序，并获取对应的值
sorted_values = torch.tensor([dict_data[key] for key in sorted(dict_data.keys())])
sorted_values1 = torch.tensor([dict_data[key]+1 for key in sorted(dict_data.keys())])
#
# 转换为张量

print(torch.sqrt(torch.sum(torch.square(sorted_values - sorted_values1))).item())



