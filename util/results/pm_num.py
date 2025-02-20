import csv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# 设置中文为黑体，英文和数字为新罗马字体
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['Times New Roman']
# 确保正常显示负号
matplotlib.rcParams['axes.unicode_minus'] = False

# tmp = 3.814697265625e-06 #0.03
# tmp = 1.9073486328125e-06 #0.05
tmp = 5.960464477539063e-08
def calculate_integer_ratio(csv_file_path):

    datanums = {
        "mrpc" : 115,
        "cola" : 268,
        "sst2" : 2105,
        "stsb" : 180,
        "rte" : 78,
        "qnli": 6547,
    }
    datanum= 0
    title = ""
    for key in datanums.keys():
        if key in csv_file_path:
            datanum = datanums[key]
            title = key

    total_numbers = 0
    integer_count = 0
    y=[]
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for item in row:
                if item:  # 确保不计算空单元格
                    float_value = float(item)
                    float_value = float_value / tmp
                    decimal_part,integer_part = math.modf(float_value)
                    y.append(decimal_part)

    start = 1 #-10 * datanum
    end = datanum * 2 + 1
    y = y[start:end]
    integer_count = y.count(0)
    total_numbers = len(y)
    if total_numbers == 0:
        return 0  # 防止除以0
    integer_ratio = integer_count / total_numbers
    print(f"整数所占的比例为: {integer_ratio}")

    y_unique = list(set(y))
    print(f"颗粒数: {len(y_unique)}")

    # 生成X轴数据，即列表的索引
    x = list(range(len(y)))
    # plt.figure(figsize=(10, 6),dpi=600)
    # fig = plt.figure(figsize=(10, 6))
    plt.scatter(x, y,s=4)
    # plt.plot(x, y)
    plt.title(f'{title}', fontsize=16)
    plt.xlabel('step',fontsize=16)
    plt.ylabel('mod(loss/min_gap)',fontsize=16)
    plt.yticks(np.linspace(0, 1, 11))
    # 显示图表
    plt.show()


def round_without_specifying_places(number):
    """
    将浮点数根据特定规则四舍五入，直到小数部分不再有连续的9。

    参数:
    number -- 要四舍五入的浮点数

    返回:
    四舍五入后的浮点数
    """
    # 将数字转换为字符串
    num_str = str(number)
    # 分离整数部分和小数部分
    integer_part, decimal_part = num_str.split('.')

    # 循环直到小数部分不再有连续的9
    if '49' in decimal_part:
        # 找到第一个'99'出现的位置
        index = decimal_part.find('49')
        # 将'99'前面的数字加1
        if index > 0:
            decimal_part = decimal_part[:index - 1] + str(5)
        else:
            integer_part = str(int(integer_part) + 1)
            decimal_part = '0'
    if '99' in decimal_part:
        # 找到第一个'99'出现的位置
        index = decimal_part.find('99')
        # 将'99'前面的数字加1
        if index > 0:
            decimal_part = decimal_part[:index - 1] + str(5)
        else:
            integer_part = str(int(integer_part) + 1)
            decimal_part = '0'

    # 将整数部分和小数部分重新组合
    rounded_number = float(integer_part + '.' + decimal_part)

    return rounded_number
def jingjian(y):
    y=list(y)
    y.sort(reverse=True)
    e=5e-9
    p=[]
    p.append(y[0])
    t=0
    for i in range(len(y)):
        if i!=0:
            if abs(y[i]-p[t])>e:
                p.append(y[i])
                t += 1
    # p=[i/0.03125 for i in p]
    p.sort()
    return p
def quchong_num(filename):

    df = pd.read_csv(filename)
    y = df['0']/tmp
    decimal_part = y%1
    # print(f'去重前：{len(y)}')
    y=decimal_part.unique()
    y=jingjian(y)
    # i=0
    # for item in y:
    #     y[i] = round_without_specifying_places(item)
    #     i += 1
    # y = np.unique(y)
    # y.sort()
    min_distance = 10000
    min_pair = (None, None)
    for i in range(len(y) - 1):
        # print(y[i])
        # 计算当前元素和下一个元素之间的距离
        distance = abs(y[i + 1] - y[i])
        # 更新最小距离
        if distance < min_distance:
            min_distance = distance
            min_pair = (y[i + 1] , y[i])
    # 打印最小距离
    # x = list(range(len(y)))
    #
    # plt.scatter(x, y, s=4)
    # # plt.plot(x, y)
    # plt.title(f'sst-2', fontsize=16)
    # plt.xlabel('step', fontsize=16)
    # plt.ylabel('mod(loss/min_gap)', fontsize=16)
    # plt.yticks(np.linspace(0, 1, 11))
    # # 显示图表
    # plt.show()
    print(f'去重后：{len(y)}')
    # print(f"列表中任意两个数的最小距离是: {min_distance} min_pair:{min_pair}")
def quchong_num1(filename):

    df = pd.read_csv(filename)
    y = df['0']/tmp
    decimal_part = y-y//1
    # print(f'去重前：{len(y)}')
    # y=decimal_part.unique()
    y=decimal_part
    # y=jingjian(y)
    # i=0
    # for item in y:
    #     y[i] = round_without_specifying_places(item)
    #     i += 1
    # y = np.unique(y)
    # y.sort()
    min_distance = 10000
    min_pair = (None, None)
    for i in range(len(y) - 1):
        print(y[i])
        # 计算当前元素和下一个元素之间的距离
        distance = abs(y[i + 1] - y[i])
        # 更新最小距离
        if distance < min_distance:
            min_distance = distance
            min_pair = (y[i + 1] , y[i])
    # 打印最小距离
    x = list(range(len(y)))

    plt.scatter(x, y, s=4)
    # plt.plot(x, y)
    plt.title(f'sst-2', fontsize=16)
    plt.xlabel('step', fontsize=16)
    plt.ylabel('mod(loss/min_gap)', fontsize=16)
    plt.yticks(np.linspace(0, 1, 11))
    # 显示图表
    plt.show()
    print(f'去重后：{len(y)}')
    print(f"列表中任意两个数的最小距离是: {min_distance} min_pair:{min_pair}")
# 定义函数
def f(x):
    return 0.1*(1-(1-x/1000)**3)

def fan(x):
    return 1.0/(0.02*x)
def showF():
    # 生成 x 的取值范围
    x = np.linspace(1, 100, 100)
    print(x)
    # 计算对应的 y 值
    y = fan(x)
    # y = f(x)

    # 绘制图像
    plt.plot(x, y)

    # 添加标题和标签
    plt.title("Graph of f(x) = 1/x")
    plt.xlabel("x")
    plt.ylabel("fan(x)")

    # 显示图像
    plt.grid(True)
    plt.show()
def CrossEntropyLoss():
    # labels = torch.tensor([0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0,
    #                        0, 1, 1, 0, 0, 0, 1, 1])
    # logits = torch.tensor([[0.3900, 0.3587],
    #                        [0.4072, 0.3517],
    #                        [0.3869, 0.3479],
    #                        [0.4075, 0.3910],
    #                        [0.3981, 0.3713],
    #                        [0.3918, 0.3312],
    #                        [0.4020, 0.4066],
    #                        [0.3950, 0.3520],
    #                        [0.3965, 0.3493],
    #                        [0.3961, 0.3745],
    #                        [0.4053, 0.3435],
    #                        [0.4033, 0.3599],
    #                        [0.4029, 0.3504],
    #                        [0.3984, 0.3641],
    #                        [0.3898, 0.3671],
    #                        [0.4040, 0.3749],
    #                        [0.4181, 0.3451],
    #                        [0.3958, 0.3713],
    #                        [0.3921, 0.3508],
    #                        [0.3922, 0.3621],
    #                        [0.4084, 0.3544],
    #                        [0.4027, 0.3492],
    #                        [0.3943, 0.3680],
    #                        [0.4010, 0.3709],
    #                        [0.4049, 0.3386],
    #                        [0.4024, 0.3640],
    #                        [0.3939, 0.3679],
    #                        [0.3925, 0.3578],
    #                        [0.4097, 0.3390],
    #                        [0.4038, 0.3994],
    #                        [0.3818, 0.3588],
    #                        [0.4067, 0.3953]])
    # logits1 = torch.tensor([[0.4067, 0.3655],
    #                 [0.4043, 0.3524],
    #                 [0.3927, 0.3704],
    #                 [0.3917, 0.3415],
    #                 [0.4002, 0.3623],
    #                 [0.3985, 0.3797],
    #                 [0.3892, 0.3597],
    #                 [0.3961, 0.3588],
    #                 [0.3984, 0.3577],
    #                 [0.3861, 0.3802],
    #                 [0.4032, 0.3711],
    #                 [0.3968, 0.3515],
    #                 [0.4380, 0.3644],
    #                 [0.4057, 0.3621],
    #                 [0.3999, 0.3585],
    #                 [0.4125, 0.3696],
    #                 [0.3995, 0.3544],
    #                 [0.4105, 0.3490],
    #                 [0.4120, 0.3458],
    #                 [0.3936, 0.3898],
    #                 [0.4012, 0.3834],
    #                 [0.3849, 0.3736],
    #                 [0.3921, 0.3444],
    #                 [0.3949, 0.3676],
    #                 [0.4087, 0.3553],
    #                 [0.3969, 0.3746],
    #                 [0.3995, 0.3638],
    #                 [0.3828, 0.3620],
    #                 [0.4077, 0.3728],
    #                 [0.4010, 0.3668],
    #                 [0.4050, 0.3642],
    #                 [0.4011, 0.3595]])
    labels = torch.tensor([0, 1, 1, 1])
    logits = torch.tensor([[0.3900, 0.3587],
                           [0.4072, 0.3517],
                           [0.3869, 0.3479],
                           [0.4075, 0.3910]])
    logits1 = torch.tensor([[0.4067, 0.3655],
                            [0.4043, 0.3524],
                            [0.3927, 0.3704],
                            [0.3917, 0.3415]])
    logits2 = torch.tensor([[1.0, 0],
                            [1.0, 0],
                            [1.0, 0],
                            [1.0, 0]])
    crossentropyloss = nn.CrossEntropyLoss()
    crossentropyloss_output = crossentropyloss(logits,labels)
    print('crossentropyloss_output:\n', crossentropyloss_output.item())#0.6964

    #等价于
    # 对比softmax与log的结合与nn.LogSoftmaxloss(负对数似然损失)的输出结果，发现两者是一致的。
    logsoftmax_func = nn.LogSoftmax(dim=1)
    logsoftmax_output = logsoftmax_func(logits)
    print('logsoftmax_output:\n', logsoftmax_output)

    # softmax_func = nn.Softmax(dim=1)
    # soft_output = softmax_func(logits)
    # logsoftmax_output = torch.log(soft_output)
    # print('logsoftmax_output:\n', logsoftmax_output)

    nlloss_output = torch.mean(-logsoftmax_output[np.arange(4),labels])
    nlloss_output1 = torch.mean(-logsoftmax_output[np.arange(4),1-labels])

    # pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
    nllloss_func = nn.NLLLoss()
    # nlloss_output = nllloss_func(logsoftmax_output, labels)
    # nlloss_output1 = nllloss_func(logsoftmax_output, 1-labels)
    print('nlloss_output:\n', nlloss_output.item(),nlloss_output1.item(),((nlloss_output+nlloss_output1)/2.0).item(),torch.mean(torch.tensor([nlloss_output,nlloss_output1])).item(),torch.mean(-logsoftmax_output).item())

def show_loss_gap_2(filenamelist):
    df = []
    fp = []
    start = 0
    for filename in filenamelist:

        data = pd.read_csv(filename)
        y = data / tmp
        decimal_part = y % 1
        fp.append(decimal_part)

    fig, axs = plt.subplots(2, 1)
    a = 255
    label = {}
    label[0] = f'The change of loss particles with the training process'
    label[1] = f'The change of loss particles with the training process'
    text=['MNLI','QQP']
    p=['(a)','(b)']
    for index, item in enumerate(fp):
        x = range(len(item))
        axs[index].scatter(x,item,color=(114 / a, 188 / a, 213 / a),s=2,label=f'{text[index]}')
        axs[index].set_title(f'{label[index]}', fontsize=18)
        axs[index].set_xticks(range(0, len(item), 2000))
        axs[index].set_yticks(np.arange(0, 1 + 0.0625, 0.125))
        axs[index].set_ylabel(rf'$L \: \% \: L_\delta$', fontsize=18)
        axs[index].legend(fontsize=18,loc="upper left")
        axs[index].text(0.5, -0.2, f'{p[index]}', transform=axs[index].transAxes, fontsize=16, ha='center', va='top')
        axs[index].set_xlabel('Step', fontsize=18)
    # 关闭网格线
    plt.grid(False)

    # plt.subplots_adjust(hspace=0.3)  # hspace控制垂直间距
    plt.subplots_adjust(wspace=0.05)  # wspace控制水平间距
    plt.tight_layout()
    # plt.suptitle('sst-2', fontsize=18)
    plt.show()

if __name__ == '__main__':
    #1-1
    file = ['实验/1-1/loss_history_mnli_0.0_3404_2_ft.csv','实验/1-1/loss_history_qqp_0.0_3404_2_ft.csv']
    # show_loss_gap_2(file)
    # a=[
    #     "颗粒/loss_history_qnli_0.0_3404_2_ft.csv",
    #     "颗粒/loss_history_qnli_0.004_3404_2_ft.csv",
    #     "颗粒/loss_history_mnli_0.0_3404_2_ft.csv",
    #     "颗粒/loss_history_mnli_0.004_3404_2_ft.csv",
    #     "颗粒/loss_history_qqp_0.0_3404_2_ft.csv",
    #     "颗粒/loss_history_qqp_0.004_3404_2_ft.csv",
    # ]
    # for i in a:
    #     print(i)
    #     csv_file_path = i
    #     quchong_num(csv_file_path)
    csv_file_path="实验/1-1/loss_history_qqp_0.0_3404_2_ft.csv"
    # quchong_num1(csv_file_path)

    # calculate_integer_ratio(csv_file_path)
    # print(2.9802322387695312e-08/1.862645149230957e-09)