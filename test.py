# from transformers import pipeline,AutoModelForSequenceClassification
import numpy as np

# 初始化数据集大小
dataset_size = 100000  # 示例数据集大小

# 初始化 score 数组，随机生成分数
score = np.random.rand(dataset_size)  # 假设分数是随机生成的

# 初始化 remain 数组，存储当前保留的样本 id
remain = np.arange(dataset_size)  # 初始时保留所有样本的 id

# 模拟多个 epoch
num_epochs = 10  # 示例运行 10 个 epoch

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}:")

    # 根据当前 remain 中的样本 id 获取对应的 score
    current_scores = score[remain]

    # 第一个 epoch 选择得分最高的前 50%
    if epoch == 0:
        # 获取前 50% 的索引
        top_50_indices = np.argsort(current_scores)[-len(remain) // 2:]
        # 更新 remain 为得分最高的前 50% 的样本 id
        remain = remain[top_50_indices]
        print(f"Selected top 50% samples. Remaining samples: {len(remain)}")
    else:
        # 后续 epoch 选择得分最高的前 90%
        top_90_indices = np.argsort(current_scores)[-int(0.9 * len(remain)):]
        # 更新 remain 为得分最高的前 90% 的样本 id
        remain = remain[top_90_indices]
        print(f"Selected top 90% samples. Remaining samples: {len(remain)}")

    # 模拟更新 score（这里假设分数会随机变化）
    score = np.random.rand(dataset_size)  # 在实际应用中，这里应该是根据模型的输出更新分数

# 最终保留的样本 id
print("Final remaining samples:", remain)
