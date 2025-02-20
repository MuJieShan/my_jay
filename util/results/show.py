import matplotlib
import matplotlib.pyplot as plt
import re
import json
import pandas as pd
import torch
# from transformers import BertTokenizer, BertModel
import os
import seaborn as sns
from tqdm import tqdm
import matplotlib.ticker as mticker
from matplotlib.ticker import MultipleLocator

# matplotlib.rcParams['font.family'] = ['Times New Roman']
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# 确保正常显示负号
# matplotlib.rcParams['axes.unicode_minus'] = False
mrpc=115#300,600
cola=268
sst2=2105#1000,4000
stsb=180
rte = 78
qnli = 3274
num=73

a=255
color = {}
color[0] = (114 / a, 188 / a, 213 / a)#浅蓝
color[1] = (255 / a, 208 / a, 111 / a)#橙色色
color[2] = (55 / a, 103 / a, 149 / a)#深蓝色
color[3] = (231 / a, 98 / a, 84 / a)#红色
color[4] = (14 / a, 135 / a, 204 / a)
color[5] = (82 / a, 143 / a, 173 / a)


# def showbert():
#     model=BertModel.from_pretrained('D:/Python/model/bert')
#     print(model)
    # print(f'model.encoder.layer.0.intermediate.dense.weight.shape: {model.encoder.layer[0].intermediate.dense.weight.shape:}')
def plot_reg_history(filename):
    # 使用os.path.basename获取文件名
    file_name_with_extension = os.path.basename(filename)

    # 使用os.path.splitext分离文件名和扩展名
    file, file_extension = os.path.splitext(file_name_with_extension)

    # 从 CSV 文件加载 reg 值
    df = pd.read_csv(filename)
    start =int(mrpc*9)
    end = int(mrpc*7)
    # 绘制 reg 的变化曲线
    plt.figure(figsize=(10, 6))
    # print(df.columns)
    query = [column for column in df.columns if 'query' in column]
    key = [column for column in df.columns if 'key' in column]
    value = [column for column in df.columns if 'value' in column]
    attoutput = [column for column in df.columns if 'attention.output.dense' in column]
    output = [column for column in df.columns if 'output.dense' in column]
    inter = [column for column in df.columns if 'intermediate' in column]

    df['mean_query'] = df[query].mean(axis=1)
    df['key'] = df[key].mean(axis=1)
    df['value'] = df[value].mean(axis=1)
    df['attoutput'] = df[attoutput].mean(axis=1)
    df['output'] = df[output].mean(axis=1)
    df['inter'] = df[inter].mean(axis=1)

    q=df['mean_query'][end:start].to_numpy()
    k=df['key'][end:start].to_numpy()
    v=df['value'][end:start].to_numpy()
    ao=df['attoutput'][end:start].to_numpy()
    o=df['output'][end:start].to_numpy()
    i=df['inter'][end:start].to_numpy()

    q_mean=q.mean()
    k_mean=k.mean()
    v_mean=v.mean()
    ao_mean=ao.mean()
    o_mean=o.mean()
    i_mean=i.mean()
    # for column in query :
    a=255.0
    plt.plot(q, label=f'query:{q_mean}',color=(30/a, 70/a, 110/a))
    # plt.plot(k, label=f'key:{k_mean}',color=(231/a, 98/a, 84/a))
    plt.plot(v, label=f'value:{v_mean}',color=(231/a, 98/a, 84/a))
    plt.plot(ao, label=f'attoutput:{ao_mean}',color=(91/a, 182/a, 71/a))
    plt.plot(o, label=f'output:{o_mean}',color=(255/a, 208/a, 111/a))
    plt.plot(i, label=f'inter:{i_mean}',color=(114/a, 188/a, 213/a))

    plt.xlabel('step')
    plt.ylabel(f'{file} Value')
    plt.title(f'{file} Value Change')
    # plt.ylabel('reg Value')
    # plt.title('reg Value Change')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(0, len(q), 20))
    plt.show()

import numpy as np
def replace_max_with_average(arr):
    for _ in range(2):
        # 找到最大值及其索引
        max_value = np.max(arr)
        max_index = np.argmax(arr)

        # 计算最大值左边5个值的平均值
        # 如果最大值索引小于5，取数组的前5个元素
        left_values = arr[max(0, max_index - 5):max_index]
        average_value = np.mean(left_values)

        # 替换最大值
        arr[max_index] = average_value
def plot_gtg_pr(pruned,remain):
    # 创建一个图形窗口和两个子图
    fig, (plt1, plt2) = plt.subplots(2, 1)  # 1行2列
    # 从 CSV 文件加载 reg 值
    df_p = pd.read_csv(pruned)
    df_r = pd.read_csv(remain)
    start =int(mrpc*9)-1
    end = 0#int(mrpc*7)
    # 绘制 reg 的变化曲线
    # plt.figure(figsize=(10, 6))
    df_p['mp'] = df_p.mean(axis=1)
    df_r['mr'] = df_r.mean(axis=1)
    p=df_p['mp'][end:start].to_numpy()
    r=df_r['mr'][end:start].to_numpy()
    # replace_max_with_average(p)
    # replace_max_with_average(r)
    step=2
    # 每隔step娶一个值
    p2 = p[::step]
    r2 = r[::step]
    # 每隔step取平均值
    p=p.reshape(-1, step).mean(axis=1)
    r=r.reshape(-1, step).mean(axis=1)

    p_mean = p.mean()
    r_mean = r.mean()
    a = 255.0
    # plt1.plot(p, label=f'pruned:{p_mean}', color=(114 / a, 188 / a, 213 / a))
    # plt1.plot(r, label=f'remain:{r_mean}', color=(231 / a, 98 / a, 84 / a))
    # plt1.xlabel('step')
    # plt1.ylabel(f'gtg_pr Value')
    # plt1.title(f'gtg_pr pruning Value Change')
    # plt1.legend()
    # plt1.grid(True)
    # plt1.xticks(range(0, len(p), 20))
    plt1.plot(p, label=f'pruned:{p_mean}', color=(114/a, 188/a, 213/a))
    plt1.plot(r, label=f'remain:{r_mean}', color=(231/a, 98/a, 84/a))
    plt1.set_xlabel('step')
    plt1.set_ylabel(f'gtg_pr Value')
    plt1.set_title(f'gtg_pr Value Change')
    plt1.legend()
    plt1.grid(True)
    plt1.set_xticks(range(0, len(p), 20))

    plt2.plot(p2, label=f'pruned:{p2.mean()}', color=(114/a, 188/a, 213/a))
    plt2.plot(r2, label=f'remain:{r2.mean()}', color=(231/a, 98/a, 84/a))
    plt2.set_xlabel('step')
    plt2.set_ylabel(f'gtg_pr Value')
    plt2.set_title(f'gtg_pr Value Change')
    plt2.legend()
    plt2.grid(True)
    plt2.set_xticks(range(end, start, 100))

    # plt.tight_layout()
    plt.show()
def show_gtg_pr(pruned,remain):
    # 从 CSV 文件加载 reg 值
    df_p = pd.read_csv(pruned)
    df_r = pd.read_csv(remain)
    start = int(mrpc*10)#600#int(sst2*6)
    end = 0#int(mrpc*20)#int(mrpc*19)
    # 绘制 reg 的变化曲线
    # plt.figure(figsize=(10, 6))
    print(df_p.shape)
    print(df_r.shape)
    df_p['mp'] = df_p.mean(axis=1)
    df_r['mr'] = df_r.mean(axis=1)
    p=df_p['mp'][end:start].to_numpy()
    r=df_r['mr'][end:start].to_numpy()
    print(f"pruned梯度为0所占的比例：{np.count_nonzero(p == 0.0)/p.size},min:{np.min(p)}")
    print(f"remain梯度为0所占的比例：{np.count_nonzero(r == 0.0)/r.size},min:{np.min(r)}")
    p_mean = p.mean()
    r_mean = r.mean()
    a = 255.0
    plt.plot(p, label=f'mrpc pruned:mean:{round(p_mean,6)},var:{round(p.var(),6)}', color=(114 / a, 188 / a, 213 / a))
    # plt.plot(r, label=f'mrpc remain:mean:{round(r_mean,6)},var:{round(r.var(),6)}', color=(231 / a, 98 / a, 84 / a))
    plt.xlabel('steps')
    plt.ylabel(f'gtg Value')
    plt.title(f'mrpc gtg Value Change')
    plt.legend()
    plt.xticks(range(0, len(p),100))
    plt.show()

def show_gtg(filename, basegtg):
    # 从 CSV 文件加载 reg 值
    fp = pd.read_csv(filename)
    bp = pd.read_csv(basegtg)
    start = -1
    end = -250
    fp['mp'] = fp.mean(axis=1)
    bp['mr'] = bp.mean(axis=1)
    f = fp['mp'][20*mrpc:29*mrpc].to_numpy()
    b = bp['mr'][900:900+mrpc*9].to_numpy()
    f_mean = f.mean()
    b_mean = b.mean()
    a = 255.0
    plt.plot(f, label=f'fp_gtg,mean:{round(f_mean,6)},var:{round(f.var(),6)}', color=(114 / a, 188 / a, 213 / a))
    plt.plot(b, label=f'base_gtg,mean:{round(b_mean,6)},var:{round(b.var(),6)}', color=(231 / a, 98 / a, 84 / a))
    plt.xlabel('step')
    plt.ylabel(f'gtg Value')
    plt.title(f'gtg Value Compare')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(0, len(f), 100))
    plt.show()
def show_loss(filename,base_loss):

    file_name_with_extension = os.path.basename(filename)
    file, file_extension = os.path.splitext(file_name_with_extension)
    df = pd.read_csv(filename)
    # df_1 = pd.read_csv(f"{file}_0.01.csv")
    # df_2 = pd.read_csv(f"{file}_0.001.csv")
    # df_3 = pd.read_csv(f"{file}_0.0001.csv")
    df1 = pd.read_csv(base_loss)
    bp = df1[cola*10:cola*12].to_numpy()
    start =0#int(mrpc*20) #30#
    end = int(mrpc*10)#600#int(sst2)#int(mrpc*30)
    fp = df[start:end].to_numpy()
    # fp1 = df_1[start:end]
    # fp2 = df_2[start:end]
    # fp3 = df_3[start:end]

    plt.figure(figsize=(10, 6))
    a=255
    # plt.plot(fp1, label=f'0.01', color=(231/a, 98/a, 84/a))
    # plt.plot(fp2, label=f'0.001', color=(255/a, 208/a, 111/a))
    # plt.plot(fp3, label=f'0.0001', color=(114/a, 188/a, 213/a))
    plt.plot(fp, label=f'mrpc loss(unfreezen),var:{round(fp.var(),6)}', color=(114 / a, 188 / a, 213 / a))
    # plt.plot(bp, label=f'cola loss(0.008),var:{round(bp.var(),6)}', color=(231/a, 98/a, 84/a))
    plt.xlabel('step')
    plt.ylabel(f'loss value')
    plt.title(f'mrpc loss Value Changed')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(0,len(fp), 20))

    # plt.tight_layout()
    plt.show()
def show_dataset_g1norm(filename):
    s=""
    if "pruned" in filename:
        s="pruned"
    if "remain" in filename:
        s="remain"
    g1 = pd.read_csv(filename).to_numpy()
    length = len(g1)
    g1=g1/num
    print(f"{filename}:len:{len(g1)} max:{max(g1)} min:{min(g1)} mean:{sum(g1)/len(g1)}")
    # 绘制直方图
    a=255
    counts, bins, patches = plt.hist(g1, bins=20, edgecolor="black", color=(82 / a, 143 / a, 173 / a))
    # 在每个柱子的顶部显示数量
    for count, x in zip(counts, bins[:-1]):
        plt.text(x + (bins[1] - bins[0]) / 2, count, str(int(count)), ha='center', va='bottom')
    # 添加标题和标签
    # plt.xlim(left=min(g1), right=max(g1))
    plt.title(f"The one-norm distribution of the {s} gradient of each data")
    plt.xlabel("one-norm of the gradient")
    plt.ylabel("Frequency")
    plt.xticks(bins,rotation=45)
    # 显示图形
    plt.show()
def show_midu_g1norm(pruned,remain):
    pruned = pd.read_csv(pruned).to_numpy().flatten()
    remain = pd.read_csv(remain).to_numpy().flatten()
    length = len(pruned)
    pruned=pruned/num
    remain=remain/num
    # 绘制直方图
    sns.kdeplot(pruned, fill=True, color="red",label='pruned',cut=0)
    sns.kdeplot(remain, fill=True, color='blue',label='remain',cut=0)
    # 添加标题和标签
    plt.legend()
    plt.title("The one-norm distribution of the gradient of each data")
    plt.xlabel("one-norm of the gradient")
    plt.ylabel("Density")
    l=(max(max(pruned),max(remain))-min(min(pruned),min(remain))+1)/20
    plt.xticks(np.arange(min(min(pruned),min(remain)),max(max(pruned),max(remain))+1,l),rotation=45)
    # 显示图形
    plt.show()
def rm_maxn(list,n):
    l = int(len(list)*n)
    return sorted(list)[:-l] if l < len(list) else []
def show_midu_g1_gap(prunedfile,remainfile):
    p = pd.read_csv(prunedfile)
    p0 = p['0']
    p1 = p['1']
    r = pd.read_csv(remainfile)#.to_numpy().flatten()
    r0 = r['0']
    r1 = r['1']

    pruned=abs(p0-p1)
    pruned = rm_maxn(pruned,0.0015)
    remain=abs(r0-r1)
    remain = rm_maxn(remain,0.0015)
    # 绘制直方图
    sns.kdeplot(pruned, fill=True, color="red",label='pruned',cut=0)
    sns.kdeplot(remain, fill=True, color='blue',label='remain',cut=0)
    # 添加标题和标签
    plt.legend()
    plt.title("The one-norm distribution of the gradient gap of batch data")
    plt.xlabel("One-norm of the gradient gap")
    plt.ylabel("Density")
    l=(max(max(pruned),max(remain))-min(min(pruned),min(remain))+1)/20
    plt.xticks(np.arange(min(min(pruned),min(remain)),max(max(pruned),max(remain))+1,l),rotation=45)
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)  # 启用科学记数法
    formatter.set_powerlimits((3, 3))  # 设定显示为1e6的数值范围
    # 设置 x 轴的刻度格式
    plt.gca().xaxis.set_major_formatter(formatter)
    # 显示图形
    plt.show()

def sandian_plot(prunedfile,remainfile):
    p = pd.read_csv(prunedfile)
    p0 = p['0']
    p1 = p['1']
    r = pd.read_csv(remainfile)  # .to_numpy().flatten()
    r0 = r['0']
    r1 = r['1']

    pruned = abs(p0 - p1)
    # pruned = rm_maxn(pruned, 0.01)
    remain = abs(r0 - r1)
    # remain = rm_maxn(remain, 0.01)
    x = list(range(len(pruned)))
    # 绘制直方图
    plt.scatter(x, pruned, color='red', label='pruned',marker='o')
    plt.scatter(x, remain, color='blue', label='remain',marker='D')
    # 添加标题和标签
    plt.legend()
    plt.title("The one-norm of the gradient gap of each data")
    plt.xlabel("data")
    plt.ylabel("gradient gap")
    # 显示图形
    plt.show()

def xiaoyangbenmidu(prunedfile,remainfile):
    p = pd.read_csv(prunedfile)
    p0 = p['0']
    p1 = p['1']
    r = pd.read_csv(remainfile)#.to_numpy().flatten()
    r0 = r['0']
    r1 = r['1']
    pruned=abs(p0-p1)
    pruned = np.array(rm_maxn(pruned, 0.25))
    remain=abs(r0-r1)
    remain = np.array(rm_maxn(remain, 0.25))
    #数据处理
    # 划分为60个区间
    bins = np.linspace(min(pruned), max(pruned), 200)
    hist, bin_edges = np.histogram(pruned, bins=bins)
    # 找到数量最多的区间
    max_index = np.argmax(hist)
    selected_interval = (bin_edges[max_index], bin_edges[max_index + 1])
    mask = (pruned >= selected_interval[0]) & (pruned < selected_interval[1])
    # 选择这个区间的pruned数据
    pruned = pruned[mask]
    # 选择对应的remain数据
    remain = remain[mask]

    # 绘制直方图
    sns.kdeplot(pruned, fill=True, color="red",label='pruned',cut=0)
    sns.kdeplot(remain, fill=True, color='blue',label='remain',cut=0)
    # 添加标题和标签
    plt.legend()
    plt.title(f"The one-norm distribution of the gradient gap of dense {len(pruned)} data")
    plt.xlabel("one-norm of the gradient gap")
    plt.ylabel("Density")
    l=(max(max(pruned),max(remain))-min(min(pruned),min(remain))+1)/20
    plt.xticks(np.arange(min(min(pruned),min(remain)),max(max(pruned),max(remain))+1,l),rotation=45)
    # formatter = mticker.ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)  # 启用科学记数法
    # formatter.set_powerlimits((5, 5))  # 设定显示为1e6的数值范围
    # # 设置 x 轴的刻度格式
    # plt.gca().xaxis.set_major_formatter(formatter)
    # 显示图形
    plt.show()

def difenbu(prunedfile,remainfile):
    p = pd.read_csv(prunedfile)
    p0 = p['0']
    p1 = p['1']
    r = pd.read_csv(remainfile)#.to_numpy().flatten()
    r0 = r['0']
    r1 = r['1']
    pruned=abs(p0-p1)
    pruned = np.array(rm_maxn(pruned, 0.1))
    remain=abs(r0-r1)
    remain = np.array(rm_maxn(remain, 0.1))
    #数据处理
    # 划分为60个区间
    bins = np.linspace(min(pruned), max(pruned), 8)
    hist, bin_edges = np.histogram(pruned, bins=bins)
    # 找到数量最多的区间
    max_index = np.argmin(hist)
    selected_interval = (bin_edges[max_index], bin_edges[max_index + 1])
    mask = (pruned >= selected_interval[0]) & (pruned < selected_interval[1])
    # 选择这个区间的pruned数据
    pruned = pruned[mask]
    # 选择对应的remain数据
    remain = remain[mask]

    # 绘制直方图
    sns.kdeplot(pruned, fill=True, color="red",label='pruned',cut=0)
    sns.kdeplot(remain, fill=True, color='blue',label='remain',cut=0)
    # 添加标题和标签
    plt.legend()
    plt.title(f"The one-norm distribution of the gradient gap of dense {len(pruned)} data")
    plt.xlabel("one-norm of the gradient gap")
    plt.ylabel("Density")
    l=(max(max(pruned),max(remain))-min(min(pruned),min(remain))+1)/20
    plt.xticks(np.arange(min(min(pruned),min(remain)),max(max(pruned),max(remain))+1,l),rotation=45)
    # formatter = mticker.ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)  # 启用科学记数法
    # formatter.set_powerlimits((5, 5))  # 设定显示为1e6的数值范围
    # # 设置 x 轴的刻度格式
    # plt.gca().xaxis.set_major_formatter(formatter)
    # 显示图形
    plt.show()
def get_last_int(s):
    # 使用正则表达式找到字符串中最后一个数字序列
    match = re.search(r'\d+$', s)
    # 如果找到数字，则转换为整数并返回，否则返回None
    return int(match.group()) if match else None
def get_last_number(s):
    # 正则表达式匹配浮点数，包括科学记数法
    pattern = r'[-+]?\d*\.\d+|\d+\.\d*|\d+e[-+]?\d+'
    matches = re.findall(pattern, s)
    # 返回最后一个匹配的浮点数，如果没有匹配则返回None
    return float(matches[-1]) if matches else None
def show_loss_(filename):

    file_name_with_extension = os.path.basename(filename)
    file, file_extension = os.path.splitext(file_name_with_extension)
    df = pd.read_csv(filename)
    start =0#int(mrpc*20) #30#
    end = int(mrpc*10)#600#int(sst2)#int(mrpc*30)
    fp = df[start:end].to_numpy()
    plt.figure(figsize=(10, 6))
    a=255
    plt.plot(fp, label=f'r={get_last_number(file)}', color=(114 / a, 188 / a, 213 / a))
    plt.xlabel('batch')
    plt.ylabel(f'loss')
    plt.title(f'mrpc loss')
    plt.legend()
    # 关闭网格线
    plt.grid(False)
    # 自动调整窗口
    plt.tight_layout()
    plt.xticks(range(0,len(fp), 50))

    # plt.tight_layout()
    plt.show()



def show_g1(filename):

    file_name_with_extension = os.path.basename(filename)
    file, file_extension = os.path.splitext(file_name_with_extension)
    df = pd.read_csv(filename)
    start =0#int(mrpc*20) #30#
    end = 100#600#int(sst2)#int(mrpc*30)
    fp = df[start:end].to_numpy()
    plt.figure(figsize=(10, 6))
    a=255
    x = list(range(1, len(fp) + 1))
    plt.plot(x,fp, label=f'mrpc', color=(114 / a, 188 / a, 213 / a))
    plt.xlabel('Loss rate(%)')
    plt.ylabel(f'The 1-norm of the gradient')
    plt.legend(fontsize=18)
    # 关闭网格线
    plt.grid(False)
    # 自动调整窗口
    plt.tight_layout()
    plt.xticks(range(1,end+1, 3))

    # plt.tight_layout()
    plt.show()
def show_lng1(filenamelist):
    df = []
    fp = []
    for filename in filenamelist:
        data = pd.read_csv(filename)
        df.append(data)
    start =0
    end = 100
    for i in df:
        data = i[start:end].to_numpy()
        fp.append(data)
        # fp.append(np.log(data))
    plt.figure(figsize=(10, 6))
    a=255
    color = {}
    color[0] = (114 / a, 188 / a, 213 / a)
    color[1] = (255 / a, 208 / a, 111 / a)
    color[2] = (231 / a, 98 / a, 84 / a)
    label = {}
    label[0] = f'sst-2'
    label[1] = f'mrpc'
    label[2] = f'rte'
    x = list(range(1, 101))
    for index,item in enumerate(fp):
        plt.plot(x,item, label=label[index], color=color[index])
    plt.xlabel('Compression ratio(%)',fontsize=16)
    plt.ylabel(f'Loss',fontsize=16)
    # plt.ylabel(f'The 1-norm of the gradient(ln)',fontsize=16)
    plt.legend(fontsize=18)
    # 关闭网格线
    plt.grid(False)
    # 自动调整窗口
    plt.tight_layout()
    plt.xticks(range(1,end+1, 3))

    # plt.tight_layout()
    plt.show()
def show_loss_gap_8(filenamelist):
    df = []
    fp = []
    for filename in filenamelist:
        data = pd.read_csv(filename)
        df.append(data)
    start = 0
    end = int(sst2)
    for i in df:
        data = i[start:end].to_numpy()
        fp.append(data)
    fig, axs = plt.subplots(4, 2,figsize=(10, 8))
    label = [5e-3,5e-4,5e-5,5e-6,5e-7,5e-8,3e-8,2e-8]
    a = 255
    k=0
    for i in range(4):
        for j in range(2):
            data = fp[k]
            axs[i,j].plot(data,color=(114 / a, 188 / a, 213 / a))
            # axs[i,j].legend(fontsize=18)
            # axs[i, j].set_title(f"r'$\alpha$'={label[k]}",fontsize=16)
            axs[i, j].set_title(rf'$\alpha={label[k]}$',fontsize=16)
            axs[i,j].set_xticks(range(0, len(data), 200))
            if k ==6 or k == 7:
                axs[i, j].set_xlabel('Batch', fontsize=16)
            if k in [0,2,4,6]:
                axs[i, j].set_ylabel(r"$\Delta L$", fontsize=16)
            k += 1
    # 关闭网格线
    plt.grid(False)
    # plt.subplots_adjust(hspace=0.3)  # hspace控制垂直间距
    plt.subplots_adjust(wspace=0.05)  # wspace控制水平间距
    plt.tight_layout()
    # plt.suptitle('sst-2', fontsize=18)
    plt.show()



def show_g1_gap_4(filenamelist):
    df = []
    fp = []
    for filename in filenamelist:
        data = pd.read_csv(filename)
        df.append(data)
    start = 0
    end = int(sst2)
    for i in df:
        data = i[start:end].to_numpy()
        fp.append(data)
    data_groups = [(fp[0], fp[1]), (fp[2], fp[3]),(fp[4], fp[5]),(fp[6], fp[7])]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    for ax, (data1, data2) in zip(axes.flat, data_groups):
        sns.kdeplot(data1, ax=ax, label='Data 1', shade=True)
        sns.kdeplot(data2, ax=ax, label='Data 2', shade=True)
        ax.set_title('Kernel Density Plot')  # 设置每张图的标题
        ax.legend()
    # 关闭网格线
    plt.grid(False)
    # plt.subplots_adjust(hspace=0.3)  # hspace控制垂直间距
    plt.subplots_adjust(wspace=0.05)  # wspace控制水平间距
    plt.tight_layout()
    # plt.suptitle('sst-2', fontsize=18)
    plt.show()
def show_loss_gap_2(filenamelist):
    df = []
    fp = []
    for filename in filenamelist[0]:

        data = pd.read_csv(filename)
        df.append(data)
    start = 0
    end = int(sst2)
    for i in df:
        data = i[start:end].to_numpy()
        fp.append(data)
    df1 = []
    fp1 = []
    for filename in filenamelist[1]:
        data = pd.read_csv(filename)
        df1.append(data)
    start = 0
    end = int(sst2)
    for i in df1:
        data = i[start:end].to_numpy()
        fp1.append(data)
    fig, axs = plt.subplots(1, 2,figsize=(10, 6))
    a = 255
    color = {}
    color[0] = (114 / a, 188 / a, 213 / a)
    color[1] = (255 / a, 208 / a, 111 / a)
    color[2] = (231 / a, 98 / a, 84 / a)
    label = {}
    label[0] = f'sst-2'
    label[1] = f'mrpc'
    label[2] = f'rte'
    x = list(range(1, 101))

    for index, item in enumerate(fp):
        axs[0].plot(x, item, label=label[index], color=color[index])
        axs[0].set_xlabel('Compression ratio(%)', fontsize=16)
        axs[0].set_ylabel(f'The 1-norm of the gradient(ln)',fontsize=16)
        axs[0].legend(fontsize=18)
    for index, item in enumerate(fp1):
        axs[1].plot(x, item, label=label[index], color=color[index])
        axs[1].set_xlabel('Compression ratio(%)', fontsize=16)
        axs[1].set_ylabel(f'Loss', fontsize=16)
        axs[1].legend(fontsize=18)
    # 关闭网格线
    plt.grid(False)
    # plt.subplots_adjust(hspace=0.3)  # hspace控制垂直间距
    plt.subplots_adjust(wspace=0.05)  # wspace控制水平间距
    plt.tight_layout()
    # plt.suptitle('sst-2', fontsize=18)
    plt.show()

def show_sparsity(filename):
    file_name_with_extension = os.path.basename(filename)
    file, file_extension = os.path.splitext(file_name_with_extension)
    df = pd.read_csv(filename)
    start = 0
    end = int(sst2)
    fp = df[start:end].to_numpy().flatten().tolist()
    zero_count = fp.count(0)
    total_count = len(fp)
    sparsity = zero_count / total_count
    print(f'sparsity: {zero_count} {total_count} {sparsity}')

def show_OneNormofweight(filename):
    file_name_with_extension = os.path.basename(filename)
    file, file_extension = os.path.splitext(file_name_with_extension)
    df = pd.read_csv(filename)
    # df = df.values.tolist()
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
        if key in filename:
            datanum = datanums[key]
            title = key
    start = 0  # int(mrpc*20) #30#
    end =  int(datanum)*3 + 1  # 600#int(sst2)#int(mrpc*30)1*768+1
    fp = df[start:].to_numpy()
    plt.figure(figsize=(10, 6))
    a = 255
    # plt.plot(fp, label=f'step={get_last_int(file)}', color=(114 / a, 188 / a, 213 / a))
    #散点图
    fp = np.sort(fp,axis=0)
    x = range(len(fp))
    plt.scatter(x,fp, label=f'r={get_last_number(file)},mean={fp.mean()}', s=5,color=(114 / a, 188 / a, 213 / a))
    # plt.plot(x,fp,label=f'r={get_last_number(file)},mean={fp.mean()}', color=(114 / a, 188 / a, 213 / a))
    # plt.xlabel('维度',fontsize=16)
    plt.xlabel('data',fontsize=16)
    plt.ylabel(f'Score',fontsize=16)
    plt.title(f'{title}',fontsize=16)
    plt.legend(fontsize=18)
    # 关闭网格线
    plt.grid(False)
    # 自动调整窗口
    plt.tight_layout()
    plt.xticks(range(0, len(fp), 2000))

    # plt.tight_layout()
    plt.show()

def show_OneNormofweight_list(filenames):
    file_name_with_extension = os.path.basename(filenames[0])
    file, file_extension = os.path.splitext(file_name_with_extension)
    files = [file for file, file_extension in [os.path.splitext(filename) for filename in filenames]]
    # df = pd.read_csv(filename1)
    df = []
    for item in filenames:
        df.append(pd.read_csv(item))
    # df = df.values.tolist()
    datanums = {
        "mrpc" : 115,
        "cola" : 268,
        "sst2" : 2105,
        "stsb" : 180,
        "rte" : 78,
        "qnli": 3274,
    }
    datanum= 0
    title = ""
    for key in datanums.keys():
        if key in file:
            datanum = datanums[key]
            title = key
    start = 0  # int(mrpc*20) #30#
    end =  int(datanum)*2 + 1  # 600#int(sst2)#int(mrpc*30)1*768+1
    # fp = df[start:end].to_numpy()
    f0=df[0][start:].to_numpy().flatten()
    f1=df[1][start:].to_numpy().flatten()
    mask = f1> f0

    print(f'{len(f0[mask])/len(f0)}')
    print(f'{(len(f0)-len(f0[mask]))/len(f0)}')
    # 使用布尔索引筛选出满足条件的元素对
    # f0 = f0[mask]
    # f1 = f1[mask]
    fp = []
    i=0
    sorted_indices = np.argsort(f0)
    fp.append(f0[sorted_indices])
    fp.append(f1[sorted_indices])
    # sorted_indices=None
    # for item in df:
    #     # fp.append(np.sort(item[start:].to_numpy(),axis=0))
    #     array = item[start:].to_numpy().flatten()
    #     if i==0:
    #         sorted_indices = np.argsort(array)
    #
    #         print(sorted_indices)
    #     fp.append(array[sorted_indices])
    #     if i==1:
    #         fp.append(array)
    #     i += 1
    plt.figure(figsize=(10, 6))
    a = 255
    # plt.plot(fp, label=f'step={get_last_int(file)}', color=(114 / a, 188 / a, 213 / a))
    #散点图
    x = range(len(fp[0]))
    # plt.scatter(x,fp, label=f'r={get_last_number(file)},mean={fp.mean()}', color=(114 / a, 188 / a, 213 / a))
    labels = {}
    labels[0] = "1"
    labels[1] = "2"
    labels[2] = "loss"

    for i,item in enumerate(fp):
        plt.scatter(x, item, label=f'{labels[i]}', s=5, color=color[i])
        # plt.plot(x,item,label=f'r={get_last_number(files[i])},mean={item.mean()}', color=color[i])
        # plt.plot(x,item,label=f'{labels[i]}', color=color[i])
    # plt.xlabel('维度',fontsize=16)
    # plt.xlabel('step',fontsize=16)
    # plt.ylabel(f'OneNormofWeight',fontsize=16)
    # plt.ylabel(f'loss',fontsize=16)
    # plt.title(f'{title}(λ={get_last_number(files[0])})',fontsize=18)
    # plt.title(f'{title}',fontsize=18)
    plt.xlabel('data', fontsize=16)
    plt.ylabel(f'Score', fontsize=16)
    plt.title(f'{title} up:{len(f0[mask])/len(f0):.4f} down:{(len(f0)-len(f0[mask]))/len(f0):.4f}', fontsize=16)
    plt.legend(fontsize=18)
    # 关闭网格线
    plt.grid(False)
    # 自动调整窗口
    plt.tight_layout()
    plt.xticks(range(0, len(fp[0]), 500),fontsize=14)
    plt.yticks(fontsize=14)

    # plt.tight_layout()
    plt.show()

def show_vectors(filenames):
    file_name_with_extension = os.path.basename(filenames[0])
    file, file_extension = os.path.splitext(file_name_with_extension)
    files = [file for file, file_extension in [os.path.splitext(filename) for filename in filenames]]
    # df = pd.read_csv(filename1)
    df = []
    for item in filenames:
        df.append(pd.read_csv(item))
    # df = df.values.tolist()
    datanums = {
        "mrpc" : 115,
        "cola" : 268,
        "sst2" : 2105,
        "stsb" : 180,
        "rte" : 78,
        "qnli": 3274,
    }
    datanum= 0
    title = ""
    labels = []
    for file in files:
        for key in datanums.keys():
            if key in file:
                datanum = datanums[key]
                title = key
                labels.append(key)
    start = 1  # int(mrpc*20) #30#
    end =  768+1  # 600#int(sst2)#int(mrpc*30)1*768+1
    # fp = df[start:end].to_numpy()
    fp = []

    keli = 5.9604645e-08#3.7252903e-09#
    for item in df:
        fp.append((item[start:end]/keli).to_numpy())
    plt.figure(figsize=(10, 6))
    a = 255
    # plt.plot(fp, label=f'step={get_last_int(file)}', color=(114 / a, 188 / a, 213 / a))
    #散点图
    x = range(len(fp[0]))
    # plt.scatter(x,fp, label=f'r={get_last_number(file)},mean={fp.mean()}', color=(114 / a, 188 / a, 213 / a))

    # labels[0] = "mrpc"
    # labels[1] = "cola"
    # labels[2] = "rte"
    # labels[3] = "sst2"
    for i,item in enumerate(fp):
        plt.scatter(x, item,label=f'{labels[i]}', color=color[i])
        print(item)
    plt.xlabel('Dimension',fontsize=16)
    plt.ylabel(f'Vector',fontsize=16)
    plt.title(f'Vector',fontsize=18)
    # plt.title(f'{title}',fontsize=18)
    plt.legend(fontsize=18)
    # 关闭网格线
    plt.grid(False)
    # 自动调整窗口
    plt.tight_layout()
    plt.xticks(range(0, len(fp[0]), 50),fontsize=14)
    plt.yticks(fontsize=14)

    plt.show()
def show_vector(filename):
    file_name_with_extension = os.path.basename(filename)
    file, file_extension = os.path.splitext(file_name_with_extension)
    df = pd.read_csv(filename)
    # df = df.values.tolist()
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
        if key in filename:
            datanum = datanums[key]
            title = key
    start = 1+768*1  # int(mrpc*20) #30#
    end =  start+768*1#int(datanum)+1#int(datanum)  # 600#int(sst2)#int(mrpc*30)1*768+1
    fp = df[start:end].to_numpy()
    plt.figure(figsize=(10, 6))
    a = 255
    # plt.plot(fp, label=f'step={get_last_int(file)}', color=(114 / a, 188 / a, 213 / a))
    #散点图
    x = range(len(fp))
    print(fp.sum())
    min = np.min(np.abs(fp[fp!=0]))
    print(min)
    max = np.max(np.abs(fp[fp != 0]))
    fp=fp/min
    max=round(max/min,6)
    r=np.sqrt(sum(np.square(fp)))
    print(f"模={r}")
    # plt.scatter(x,fp, label=f'r={get_last_number(file)},mean={fp.mean()}', color=(114 / a, 188 / a, 213 / a))
    plt.scatter(x,fp, color=(114 / a, 188 / a, 213 / a),s=5)
    # plt.xlabel('维度',fontsize=16)
    plt.xlabel('Dimension',fontsize=16)
    # plt.xlabel('Batch',fontsize=16)
    plt.ylabel(f'Loss gap',fontsize=16)
    plt.title(f'{title} r={get_last_number(file)} max={max}',fontsize=16)
    # plt.legend(fontsize=18)
    # 关闭网格线
    plt.grid(False)
    # 自动调整窗口
    plt.tight_layout()
    xticks=plt.xticks(range(0, len(fp), 50))#[0]
    # for tick in xticks:
    #     plt.axvline(x=tick.get_loc(), color='gray', linestyle='--', linewidth=0.5)
    # plt.tight_layout()
    plt.show()

def show_loss_gap(filename):
    """
    散点图，损失颗粒图
    Args:
        filename:

    Returns:

    """
    file_name_with_extension = os.path.basename(filename)
    file, file_extension = os.path.splitext(file_name_with_extension)
    df = pd.read_csv(filename)
    start = 0  # int(mrpc*20) #30#
    end = int(sst2)  # 600#int(sst2)#int(mrpc*30)
    fp = df[start:].to_numpy()
    plt.figure(figsize=(10, 6))
    a = 255
    x = range(len(fp))
    # plt.plot(fp, label=f'step={get_last_int(file)}', color=(114 / a, 188 / a, 213 / a))
    # plt.plot(fp, label=f'r={get_last_number(file)}', color=(114 / a, 188 / a, 213 / a))
    plt.scatter(x, fp, label=f'MRPC',color=(114 / a, 188 / a, 213 / a), s=5)
    plt.xlabel('Single Sample',fontsize=16)
    plt.ylabel(f'Loss gap',fontsize=16)
    plt.xticks(rotation=45)
    plt.title(f'Loss Particles (BERT)',fontsize=16)
    plt.legend(fontsize=18,loc="upper right")
    # 关闭网格线
    plt.grid(False)
    # 自动调整窗口
    plt.tight_layout()
    plt.xticks(range(0, len(fp), 100))

    # plt.tight_layout()
    plt.show()
def show_loss_gap_(filename,label_file):
    """
    散点图，32样本768维度颗粒图
    Args:
        filename:

    Returns:

    """
    file_name_with_extension = os.path.basename(filename)
    file, file_extension = os.path.splitext(file_name_with_extension)
    df = pd.read_csv(filename)
    la = pd.read_csv(label_file)
    la = la[:20].to_numpy()
    print(la)
    # df = df.values.tolist()
    datanums = {
        "mrpc": 115,
        "cola": 268,
        "sst2": 2105,
        "stsb": 180,
        "rte": 78,
        "qnli": 6547,
    }
    datanum = 0
    title = ""
    for key in datanums.keys():
        if key in filename:
            datanum = datanums[key]
            title = key
    start = 0  # int(mrpc*20) #30#
    end = start + 768 * 20  # int(datanum)+1#int(datanum)  # 600#int(sst2)#int(mrpc*30)1*768+1
    fp = df[start:end].to_numpy()
    plt.figure(figsize=(10, 6))
    a = 255
    # plt.plot(fp, label=f'step={get_last_int(file)}', color=(114 / a, 188 / a, 213 / a))
    # 散点图
    x = range(len(fp))
    print(fp.sum())
    min = np.min(np.abs(fp[fp != 0]))
    print(min)
    # plt.scatter(x,fp, label=f'r={get_last_number(file)},mean={fp.mean()}', color=(114 / a, 188 / a, 213 / a))
    plt.scatter(x, fp, color=color[5], s=3)
    # plt.xlabel('维度',fontsize=16)
    plt.xlabel('Dimension', fontsize=16)
    # plt.xlabel('Batch',fontsize=16)
    plt.ylabel(f'Output gap', fontsize=16)
    plt.title(f'{title} r={get_last_number(file)} min={min}', fontsize=16)
    # plt.legend(fontsize=18)
    # 关闭网格线
    plt.grid(False)
    # 自动调整窗口
    plt.tight_layout()
    # xticks = plt.xticks(range(0, len(fp), 768))[0]
    # for tick in xticks:
    #     plt.axvline(x=tick.get_loc(), color='gray', linestyle='--', linewidth=0.5)
    p = range(384, len(fp) + 1, 768)
    x_label = [f"{i}" for i in la]
    xtick = plt.xticks(ticks=p, labels=x_label, rotation=0)
    t=0
    print(p)
    for tick in p:
        t = tick
        plt.axvline(x=t - 384, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(x=t + 384, color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def show_loss_gaps_(filenames):
    """
    散点图，32样本768维度颗粒图，及其前后变化
    Args:
        filenames:

    Returns:

    """
    file_name_with_extension = os.path.basename(filenames[0])
    file, file_extension = os.path.splitext(file_name_with_extension)
    files = [file for file, file_extension in [os.path.splitext(filename) for filename in filenames]]
    # df = pd.read_csv(filename1)
    df = []
    for item in filenames:
        df.append(pd.read_csv(item))
    datanums = {
        "mrpc": 115,
        "cola": 268,
        "sst2": 2105,
        "stsb": 180,
        "rte": 78,
        "qnli": 6547,
    }
    datanum = 0
    title = ""

    for file in files:
        for key in datanums.keys():
            if key in file:
                datanum = datanums[key]
                title = key
    start =768 * 0  # int(mrpc*20) #30#
    end = start + 768 * 32  # int(datanum)+1#int(datanum)  # 600#int(sst2)#int(mrpc*30)1*768+1
    fp = []
    for item in df:
        fp.append(item[start:].to_numpy())
    plt.figure(figsize=(10, 6))
    a = 255
    # plt.plot(fp, label=f'step={get_last_int(file)}', color=(114 / a, 188 / a, 213 / a))
    # 散点图
    x = range(len(fp[0]))
    labels = {}
    labels[0] = "before train"
    labels[1] = "after train"

    labels[2] = ""
    for i,item in enumerate(fp):
        plt.scatter(x, item,label=f'{labels[i]} 1Norm={sum(abs(item))[0]}', color=color[i],s=5)
        print(sum(abs(item))[0])
    # plt.xlabel('维度',fontsize=16)
    plt.xlabel('Dimension', fontsize=16)
    # plt.xlabel('Batch',fontsize=16)
    plt.ylabel(f'Loss gap', fontsize=16)
    plt.title(f'{title} r={get_last_number(file)}', fontsize=16)
    plt.legend(fontsize=18)
    # 关闭网格线
    plt.grid(False)
    # 自动调整窗口
    plt.tight_layout()
    p = range(384, len(fp[0]) + 1, 768)
    xtick=plt.xticks(p)

    x_label = [f"Sample {i}" for i in range(1, len(xtick) + 1)]
    plt.xticks(ticks=xtick, labels=x_label, rotation=30)
    for tick in xtick:
        t = tick
        plt.axvline(x=t - 384, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(x=t + 384, color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
def show_loss_gap_4(filenamelist):
    df = []
    fp = []
    for filename in filenamelist:
        data = pd.read_csv(filename)
        df.append(data)
    start = 0
    end = int(sst2)
    for i in df:
        data = i[start:end].to_numpy()
        fp.append(data)
    fig, axs = plt.subplots(2, 2,figsize=(10, 8))
    label = ["BERT","RoBERTA","GPT2","T5"]
    a = 255
    k=0
    for i in range(2):
        for j in range(2):
            data = fp[k]
            x = range(len(data))
            axs[i,j].scatter(x,data,color=(114 / a, 188 / a, 213 / a),s=3)
            # axs[i,j].plot(data,color=(114 / a, 188 / a, 213 / a))
            # axs[i,j].legend(fontsize=18)
            axs[i, j].set_title(f'{label[k]}',fontsize=16)
            axs[i,j].set_xticks(range(0, len(data), 200))
            if k ==2 or k == 3:
                axs[i, j].set_xlabel('Batch', fontsize=16)
            k += 1
    # 关闭网格线
    plt.grid(False)
    # plt.subplots_adjust(hspace=0.3)  # hspace控制垂直间距
    plt.subplots_adjust(wspace=0.05)  # wspace控制水平间距
    plt.tight_layout()
    # plt.suptitle('sst-2', fontsize=18)
    plt.show()
def show_loss_gap_mean(filename):
    file_name_with_extension = os.path.basename(filename)
    file, file_extension = os.path.splitext(file_name_with_extension)
    df = pd.read_csv(filename)
    # df = df.values.tolist()
    datanums = {
        "mrpc": 115,
        "cola": 268,
        "sst2": 2105,
        "stsb": 180,
        "rte": 78,
        "qnli": 6547,
    }
    datanum = 0
    title = ""
    for key in datanums.keys():
        if key in filename:
            datanum = datanums[key]
            title = key
    start = 0  # int(mrpc*20) #30#
    end = start + 768 * 32  # int(datanum)+1#int(datanum)  # 600#int(sst2)#int(mrpc*30)1*768+1
    fp = df[start:end].to_numpy()
    print(len(fp))
    plt.figure(figsize=(10, 6))
    a = 255
    # plt.plot(fp, label=f'step={get_last_int(file)}', color=(114 / a, 188 / a, 213 / a))
    # 散点图


    fp = np.array(fp.reshape(-1,768)).mean(axis=0) * (1e+7)
    print(len(fp))
    print(fp.sum())
    min = np.min(np.abs(fp[fp != 0]))
    print(min)
    l2=np.linalg.norm(fp)
    x = range(len(fp))
    # plt.scatter(x,fp, label=f'r={get_last_number(file)},mean={fp.mean()}', color=(114 / a, 188 / a, 213 / a))
    plt.scatter(x, fp, color=(114 / a, 188 / a, 213 / a), s=5)
    # plt.xlabel('维度',fontsize=16)
    plt.xlabel('Dimension', fontsize=16)
    # plt.xlabel('Batch',fontsize=16)
    plt.ylabel(f'Loss gap(*1e-7)', fontsize=16)
    plt.title(f'{title} r={get_last_number(file)} length={l2}', fontsize=16)
    # plt.legend(fontsize=18)
    # 关闭网格线
    plt.grid(False)
    # 自动调整窗口
    plt.tight_layout()
    xticks = plt.xticks(range(0, len(fp), 50))
    # for tick in xticks:
    #     plt.axvline(x=tick.get_loc(), color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
def show_acc(filename,filename2):
    file_name_with_extension = os.path.basename(filename)
    file, file_extension = os.path.splitext(file_name_with_extension)
    df = pd.read_csv(filename)
    df2 = pd.read_csv(filename2)
    # df = df.values.tolist()
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
        if key in filename:
            datanum = datanums[key]
            title = key
    fp = df[0:].to_numpy()
    l=2
    fp = fp[:(len(fp) // l) * l]
    fp = fp.reshape(-1, l).mean(axis=1)
    fp2 = df2[0:].to_numpy()
    fp2 = fp2[:(len(fp2) // l) * l]
    fp2 = fp2.reshape(-1, l).mean(axis=1)
    plt.figure(figsize=(10, 6))
    a = 255
    # plt.plot(fp, label=f'step={get_last_int(file)}', color=(114 / a, 188 / a, 213 / a))
    #散点图
    x = range(len(fp))
    print(fp.sum())
    min = np.min(np.abs(fp[fp!=0]))
    print(min)
    # plt.scatter(x,fp, label=f'r={get_last_number(file)},mean={fp.mean()}', color=(114 / a, 188 / a, 213 / a))
    plt.plot(x,fp, label="unlabel",color=(114 / a, 188 / a, 213 / a))
    plt.plot(x,fp2, label="ft",color=(255 / a, 208 / a, 111 / a))
    # plt.xlabel('维度',fontsize=16)
    plt.xlabel(f'step(*{l})',fontsize=16)
    # plt.xlabel('Batch',fontsize=16)
    plt.ylabel(f'ACC',fontsize=16)
    plt.title(f'{title}',fontsize=16)
    # plt.title(f'{title} r={get_last_number(file)}',fontsize=16)
    plt.legend(fontsize=18)
    # 关闭网格线
    plt.grid(False)
    # 自动调整窗口
    plt.tight_layout()
    xticks=plt.xticks(range(0, len(fp), 50))#[0]
    # for tick in xticks:
    #     plt.axvline(x=tick.get_loc(), color='gray', linestyle='--', linewidth=0.5)
    # plt.tight_layout()
    plt.show()
def show_s1(filename):
    # file_name_with_extension = os.path.basename(filename[0])
    # file, file_extension = os.path.splitext(file_name_with_extension)
    fp = [[], []]
    i = 0
    for file in filename:
        for f in file:
            df = pd.read_csv(f)
            fp[i].append(np.sort(df[0:].to_numpy().flatten()))
        i += 1
    # df = df.values.tolist()
    datanums = {
        "mrpc" : 115,
        "cola" : 268,
        "sst2" : 2105,
        "stsb" : 180,
        "rte" : 78,
        "qnli": 6547,
    }
    # plt.figure(figsize=(10, 6))
    a = 255
    color = {}
    color[0] = (114 / a, 188 / a, 213 / a)  # 浅蓝
    color[1] = (255 / a, 208 / a, 111 / a)  # 橙色色
    color[2] = (55 / a, 103 / a, 149 / a)  # 深蓝色
    color[3] = (231 / a, 98 / a, 84 / a)  # 红色
    color[4] = (14 / a, 135 / a, 204 / a)
    color[5] = (82 / a, 143 / a, 173 / a)
    label = ["BERT","RoBERTA","GPT2","T5"]
    marker = ['o','s','d','*']
    title = ['QNLI','MNLI']
    xticks =[5000,20000]
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    for i,f in enumerate(fp):
        x = range(len(f[0]))
        print(len(f[0]))
        for j,t in enumerate(f):
            # axs[i].scatter(x, np.log(t)+12, label=label[j],s=1, marker=marker[j],color=color[j])
            axs[i].plot(x, np.log(t)+12, label=label[j], color=color[j])
        axs[i].set_xlabel('Samples', fontsize=16)
        axs[i].set_ylabel(f'Log(Score) + C', fontsize=16)
        axs[i].set_title(f'{title[i]}', fontsize=16)
        axs[i].legend(fontsize=18)
        p=range(0, len(f[0]), xticks[i])
        axs[i].set_xticks(p)
        axs[i].set_xticklabels(p, rotation=45)
    # 关闭网格线
    plt.grid(False)
    # 自动调整窗口
    plt.tight_layout()
    plt.show()
def show_s1_xiang(filename):
    fp = [[],[]]
    i=0
    for file in filename:
        for f in file:
            df = pd.read_csv(f)
            print(df[0:].to_numpy().flatten().shape)
            fp[i].append(df[:-20].to_numpy().flatten())
        i+=1
    a = 255
    color = {}
    color[0] = (252 / a, 255 / a, 245 / a)
    color[1] = (209 / a, 219 / a, 189 / a)
    color[2] = (141 / a, 188 / a, 213 / a)
    color[3] = (55 / a, 103 / a, 149 / a)
    colors = [color[0], color[1], color[2], color[3]]
    # 创建图形和子图
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    flierprops = dict(marker='o', markerfacecolor=(231 / a, 98 / a, 84 / a), markersize=6, markeredgewidth=0)
    medianprops = dict(linestyle='-', linewidth=1, color=(231 / a, 98 / a, 84 / a))
    i=0
    for f in fp:
        outliers=[]
        for t in f:
            Q1 = np.percentile(t, 25)
            Q3 = np.percentile(t, 75)
            IQR = Q3 - Q1

            # 确定异常值的界限
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            selected_data = t[(t<lower_bound)|(t>upper_bound)]
            if len(selected_data) >500:
                selected_data = np.sort(selected_data)[:100]
            outliers.append(selected_data)
        axes[i].set_title('ResNet18-CIFAR10')
        box = axes[i].boxplot(f, patch_artist=True, medianprops=medianprops,showfliers=False)
        axes[i].set_xticklabels(['MI', 'InfoBatch', 'Greedy', 'UCB'])
        axes[i].set_ylabel('Times')
        for patch, c in zip(box['boxes'], colors):
            patch.set_facecolor(c)
        for j,outlier in enumerate(outliers):
            x_positions = [j+1 for _ in range(len(outlier))]
            # 添加异常值
            axes[i].scatter(x_positions, outlier, color='r', s=5, zorder=3, label='Outliers' if j == 0 else "")

        i+=1
    # 显示图表
    plt.tight_layout()
    plt.show()
def show_loss_gaps_2(filename):
    """
    散点图，32样本768维度颗粒图，及其前后变化
    Args:
        filenames:

    Returns:

    """
    # df = pd.read_csv(filename1)
    fp = [[], []]
    i = 0
    start =0  # int(mrpc*20) #30#
    end = start + 768 * 20  # int(datanum)+1#int(datanum)  # 600#int(sst2)#int(mrpc*30)1*768+1
    for file in filename:
        for f in file:
            df = pd.read_csv(f)
            fp[i].append(df[start:end].to_numpy().flatten())
        i += 1
    a = 255
    # colors = [[(255 / a, 208 / a, 111 / a),(114 / a, 188 / a, 213 / a)],[(114 / a, 188 / a, 213 / a),(255 / a, 208 / a, 111 / a)]]
    colors = [[(255 / a, 208 / a, 111 / a),(114 / a, 188 / a, 213 / a)],[(255 / a, 208 / a, 111 / a),(114 / a, 188 / a, 213 / a)]]
    # color[0] = (114 / a, 188 / a, 213 / a)  # 浅蓝
    # color[1] = (255 / a, 208 / a, 111 / a)  # 橙色色
    labels = [["After fine tuning", "Before fine tuning"],["Before fine tuning", "After fine tuning"]]
    # labels = [["compress", "expand"],["compress", "expand"]]
    marker = ['o', 's', 'd', '*']
    title = ['MRPC', 'SST-2']
    # title = ['MNLI', 'MRPC']
    text = ['(a)','(b)']
    xticks = [768, 768]
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    for (i, f),label,color in zip(enumerate(fp),labels,colors):
        x = range(len(f[0]))
        print(len(f[0]))
        for j, t in enumerate(f):
            # axs[i].scatter(x, np.log(t)+12, label=label[j],s=3, marker=marker[j],color=color[j])
            axs[i].scatter(x, t,label=f'{label[j]}', color=color[j],s=2)
        # axs[i].set_xlabel('Dimension', fontsize=16)
        axs[i].set_ylabel(f'Output particles', fontsize=12)
        axs[i].set_title(f'{title[i]} (768 Dimensions)', fontsize=16)
        axs[i].set_xlabel(f'{text[i]}', fontsize=12)
        axs[i].legend(fontsize=16,loc='upper right')
        p = range(384, len(f[0])+1, xticks[i])
        axs[i].set_xticks(p)
        xtick = axs[i].get_xticks()
        x_label = [f"Sample {i}" for i in range(1,len(xtick)+1)]
        axs[i].set_xticklabels(x_label, rotation=30)
        for tick in xtick:
            t=tick
            axs[i].axvline(x=t-384, color='gray', linestyle='--', linewidth=0.5)
        axs[i].axvline(x=t + 384, color='gray', linestyle='--', linewidth=0.5)
    # 关闭网格线
    plt.grid(False)
    # 自动调整窗口
    plt.tight_layout()
    plt.show()
def show_loss_acc(filename):
    fp = [[], [],[]]
    xp = []
    fp1 = [[], [],[]]
    xp1 = []
    i = 0
    for file in filename[0]:
        for f in file:
            df = pd.read_csv(f)
            fp[i].append(df['Value'].to_numpy())
        xp.append(df['Step'].to_numpy())
        i += 1
    i = 0
    for file in filename[1]:
        for f in file:
            df = pd.read_csv(f)
            fp1[i].append(df['Value'].to_numpy())
        xp1.append(df['Step'].to_numpy())
        i += 1
    a = 255
    color = {}
    color[0] = (114 / a, 188 / a, 213 / a)  # 浅蓝
    color[1] = (255 / a, 208 / a, 111 / a)  # 橙色色
    color[2] = (55 / a, 103 / a, 149 / a)  # 深蓝色
    color[3] = (231 / a, 98 / a, 84 / a)  # 红色
    color[4] = (14 / a, 135 / a, 204 / a)
    color[5] = (82 / a, 143 / a, 173 / a)
    label = ["50% high-score set", "50% low-score set"]
    line = ['-','--']
    title = ['Eval Accuracy', 'Eval Loss','Train Loss']
    text = ['(a)','(b)','(c)']
    xticks = [5000, 20000]
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    for i, f in enumerate(fp):
        x = range(len(f[0]))
        print(len(f[0]),len(xp[i]))
        for j, t in enumerate(f):
            # axs[i].scatter(x, np.log(t)+12, label=label[j],s=1, marker=marker[j],color=color[j])
            axs[0][i].plot(x, t, label=label[j], color=color[j],linestyle=line[j])
        axs[0][i].set_xlabel('Step', fontsize=12)
        # axs[i].set_ylabel(f'Log(Score) + C', fontsize=16)
        axs[0][i].set_title(f'{title[i]}', fontsize=16)
        if i==2:
          axs[0][i].legend(fontsize=12)
        if i==2:
            p = np.arange(0, len(f[0]),60)
            axs[0][i].set_xticks(p)
            axs[0][i].set_xticklabels([v for v in xp[i][p]])
        else:
            p = range(0, len(f[0]))
            axs[0][i].set_xticks(p)
            axs[0][i].set_xticklabels(xp[i])
        # axs[0][i].text(0.5, -0.16, text[i], transform=axs[0][i].transAxes, fontsize=16, ha='center',va='top')
    for i, f in enumerate(fp1):
        x = range(len(f[0]))
        print(len(f[0]),len(xp1[i]))
        for j, t in enumerate(f):
            # axs[i].scatter(x, np.log(t)+12, label=label[j],s=1, marker=marker[j],color=color[j])
            axs[1][i].plot(x, t, label=label[j], color=color[j],linestyle=line[j])
        axs[1][i].set_xlabel('Step', fontsize=12)
        # axs[i].set_ylabel(f'Log(Score) + C', fontsize=16)
        axs[1][i].set_title(f'{title[i]}', fontsize=16)
        if i == 2:
            axs[1][i].legend(fontsize=12)
        if i==2:
            p = np.arange(0, len(f[0]),60)
            axs[1][i].set_xticks(p)
            axs[1][i].set_xticklabels([v for v in xp1[i][p]])
        else:
            p = range(0, len(f[0]))
            axs[1][i].set_xticks(p)
            axs[1][i].set_xticklabels(xp1[i])
        # axs[1][i].text(0.5, -0.2, text[i], transform=axs[1][i].transAxes, fontsize=16, ha='center',va='top')
    fig.text(0.5, 0.5, '(a) MNLI', ha='center',fontsize=16)
    fig.text(0.5, 0.02, '(b) QQP', ha='center',fontsize=16)
    # 关闭网格线
    plt.grid(False)
    # 自动调整窗口
    plt.tight_layout()
    plt.show()
def show_lossgap_poolergap(filenamelist):
    df = []
    fp = []
    for filename in filenamelist[0]:

        data = pd.read_csv(filename)
        df.append(data)
    start = 0
    end = int(sst2)
    for i in df:
        data = i[start:].to_numpy()
        fp.append(data)
    df1 = []
    fp1 = []
    for filename in filenamelist[1]:
        data = pd.read_csv(filename)
        df1.append(data)
    start = 0
    end = int(sst2)
    for i in df1:
        data = i[start:768].to_numpy()
        fp1.append(data)
    fig, axs = plt.subplots(1, 2,figsize=(10, 6))
    a = 255
    color = {}
    color[0] = (114 / a, 188 / a, 213 / a)
    color[1] = (255 / a, 208 / a, 111 / a)
    color[2] = (231 / a, 98 / a, 84 / a)
    label = {}
    label[0] = f'sst-2'
    label[1] = f'mrpc'
    label[2] = f'rte'
    # x = list(range(1, 101))

    for index, item in enumerate(fp):
        x=range(len(item))
        axs[0].scatter(x, item, color=color[0], label="SST-2",s=3)
        axs[0].set_xlabel('Batch', fontsize=16)
        axs[0].set_ylabel(f'Loss Gap',fontsize=16)
        axs[0].set_title(f'Loss Particles', fontsize=18)
        axs[0].legend(fontsize=14,loc='upper right')
        axs[0].text(0.5, -0.12, '(a)', transform=axs[0].transAxes, fontsize=16, ha='center', va='top')
    for index, item in enumerate(fp1):
        x = range(len(item))
        axs[1].scatter(x, item, color=color[0],label="MRPC", s=3)
        axs[1].set_xlabel('Dimension', fontsize=16)
        axs[1].set_ylabel(f'Output Gap', fontsize=16)
        axs[1].set_title(f'Output Particles', fontsize=18)
        axs[1].legend(fontsize=14,loc='upper right')
        axs[1].text(0.5, -0.12, '(b)', transform=axs[1].transAxes, fontsize=16, ha='center', va='top')
    # 关闭网格线
    plt.grid(False)
    # plt.subplots_adjust(hspace=0.3)  # hspace控制垂直间距
    plt.subplots_adjust(wspace=0.05)  # wspace控制水平间距
    plt.tight_layout()
    # plt.suptitle('sst-2', fontsize=18)
    plt.show()
def show_pooler_2_3(filenamelist):
    df = []
    fp = []
    for filename in filenamelist:
        data = pd.read_csv(filename)
        df.append(data)
    start = 0
    end = 768
    for i in df:
        data = i[start:end].to_numpy()
        fp.append(data)
    fig, axs = plt.subplots(2, 3,figsize=(10, 6))
    label = [rf"$\alpha=2e-8$",rf"$\alpha=3e-8 \textasciitilde 8e-8$",rf"$\alpha=9e-8,1e-7$",rf"$\alpha=2e-8$",rf"$\alpha=3e-8 \textasciitilde 8e-8$",rf"$\alpha=9e-8,1e-7$"]
    text = ['(a) BERT without fine-tuning','(b) BERT with fine-tuning']
    a = 255
    k=0
    for i in range(2):
        for j in range(3):
            data = fp[k]
            p=[abs(x) for x in data if x != 0]
            if len(p)!=0:
                min_ = min(p).item()
                min_1 = rf'$h_\delta={min_: .6e}$'
            else:
                min_1 = rf'$h_\delta=0$'
            sum_ = sum(abs(data)).item()
            x = range(len(data))
            axs[i,j].scatter(x,data,color=(114 / a, 188 / a, 213 / a),label="MRPC",s=3)
            axs[i,j].set_xlabel(f'{label[k]}', fontsize=14)
            axs[i,j].set_ylabel(f'Output Gap', fontsize=14)
            axs[i,j].set_title(f'Output Particles({min_1})', fontsize=18)
            axs[i,j].legend(fontsize=14, loc='upper right')
            axs[i,1].text(0.5, -0.2, f'{text[i]}', transform=axs[i,1].transAxes, fontsize=16, ha='center', va='top')
            k += 1
    # 关闭网格线
    plt.grid(False)
    # plt.subplots_adjust(hspace=0.3)  # hspace控制垂直间距
    plt.subplots_adjust(wspace=0.05)  # wspace控制水平间距
    plt.tight_layout()
    # plt.suptitle('sst-2', fontsize=18)
    plt.show()

def show_pooler_2_4(filenamelist):
    df = []
    fp = []
    for filename in filenamelist:
        data = pd.read_csv(filename)
        df.append(data)
    start = 0
    end = 768*5
    for i in df:
        data = i[start:end].to_numpy()
        fp.append(data)
    fig, axs = plt.subplots(2, 2,figsize=(10, 6))
    label = [rf"$\alpha=2e-8$",rf"$\alpha=3e-8 \textasciitilde 8e-8$",rf"$\alpha=9e-8,1e-7$",rf"$\alpha=2e-8$",rf"$\alpha=3e-8 \textasciitilde 8e-8$",rf"$\alpha=9e-8,1e-7$"]
    text = ['(a) BERT without fine-tuning','(b) BERT with fine-tuning']
    a = 255
    k=0
    for i in range(2):
        for j in range(2):
            data = fp[k]
            p=[abs(x) for x in data if x != 0]
            if len(p)!=0:
                min_ = min(p).item()
                min_1 = rf'$h_\delta={min_: .6e}$'
            else:
                min_1 = rf'$h_\delta=0$'
            sum_ = sum(abs(data)).item()
            x = range(len(data))
            axs[i,j].scatter(x,data,color=(114 / a, 188 / a, 213 / a),label="MRPC",s=3)
            axs[i,j].set_xlabel(f'{label[k]}', fontsize=14)
            axs[i,j].set_ylabel(f'Output Gap', fontsize=14)
            axs[i,j].set_title(f'Output Particles({min_1})', fontsize=18)
            axs[i,j].legend(fontsize=14, loc='upper right')
            # axs[i,1].text(0.5, -0.2, f'{text[i]}', transform=axs[i,1].transAxes, fontsize=16, ha='center', va='top')
            k += 1

    # 关闭网格线
    plt.grid(False)
    # plt.subplots_adjust(hspace=0.3)  # hspace控制垂直间距
    plt.subplots_adjust(wspace=0.05)  # wspace控制水平间距
    plt.tight_layout()
    # plt.suptitle('sst-2', fontsize=18)
    plt.show()
def show_pooler_2_5(filenamelist):
    df = []
    fp = []
    for filename in filenamelist:
        data = pd.read_csv(filename)
        df.append(data)
    start = 0
    end = 768
    for i in df:
        data = i[start:end].to_numpy()
        fp.append(data)
    fig, axs = plt.subplots(1, 3,figsize=(10, 6))
    label = [rf"(a) $\alpha=5e-8$",rf"(b) $\alpha=6e-8 \textasciitilde 1e-7$",rf"(c) $\alpha=2e-7$",rf"$\alpha=2e-8$",rf"$\alpha=3e-8 \textasciitilde 8e-8$",rf"(c) $\alpha=9e-8,1e-7$"]
    text = [rf'Output particles under slightly expanded $\theta$']
    a = 255
    k=0
    for i in range(3):
        data = fp[k]
        p=[abs(x) for x in data if x != 0]
        if len(p)!=0:
            min_ = min(p).item()
            min_1 = rf'$h_\delta={min_: .6e}$'
        else:
            min_1 = rf'$h_\delta=0$'
        sum_ = sum(abs(data)).item()
        print(len(data))
        x = range(len(data))
        axs[i].scatter(x,data,color=(114 / a, 188 / a, 213 / a),label="MRPC",s=3)
        axs[i].set_xlabel(f'{label[k]}', fontsize=14)
        axs[i].set_ylabel(f'Output Gap', fontsize=14)
        axs[i].set_title(f'{min_1}', fontsize=18)
        axs[i].legend(fontsize=14, loc='upper right')
        # axs[i].text(0.5, 0.2, f'{text[0]}', transform=axs[1].transAxes, fontsize=16, ha='center', va='top')
        k += 1
    # 关闭网格线
    fig.text(0.5, 0.95, f'{text[0]}', ha='center', fontsize=18)
    plt.grid(False)
    # plt.subplots_adjust(hspace=0.3)  # hspace控制垂直间距
    plt.subplots_adjust(wspace=0.05)  # wspace控制水平间距
    plt.tight_layout()
    # plt.suptitle('sst-2', fontsize=18)
    plt.show()
def show_lossgap_1_8(filenamelist):
    df = []
    fp = []
    for filename in filenamelist:
        data = pd.read_csv(filename)
        df.append(data)
    start = 0
    end = int(sst2)
    for i in df:
        data = i[start:].to_numpy()
        fp.append(data)

    fig, axs = plt.subplots(1, 2,figsize=(10, 6))
    a = 255
    color = {}
    color[0] = (114 / a, 188 / a, 213 / a)
    color[1] = (255 / a, 208 / a, 111 / a)
    color[2] = (231 / a, 98 / a, 84 / a)
    label = {}
    label[0] = f'SST-2'
    label[1] = f'STS-B'
    text = ['(a)','(b)']
    # x = list(range(1, 101))

    for i in range(2):
        item = fp[i]
        p = [abs(x) for x in item if x != 0]
        if len(p) != 0:
            min_ = min(p).item()
            min_1 = rf'$L_\delta={min_: .6e}$'
        else:
            min_1 = rf'$L_\delta=0$'
        x=range(len(item))
        axs[i].scatter(x, item, color=color[0], label=f"{label[i]}",s=5)
        axs[i].set_xlabel('Batch', fontsize=16)
        axs[i].set_ylabel(f'Loss Gap',fontsize=16)
        axs[i].set_title(f'Loss Particles({min_1})', fontsize=18)
        axs[i].legend(fontsize=14,loc='upper right')
        axs[i].text(0.5, -0.12, f'{text[i]}', transform=axs[i].transAxes, fontsize=16, ha='center', va='top')
    # 关闭网格线
    plt.grid(False)
    # plt.subplots_adjust(hspace=0.3)  # hspace控制垂直间距
    plt.subplots_adjust(wspace=0.05)  # wspace控制水平间距
    plt.tight_layout()
    # plt.suptitle('sst-2', fontsize=18)
    plt.show()
if __name__ == '__main__':
    #1-2
    file=['实验/1-2/bert-loss_gap_sst2_0.0_5e-08_3404.csv','实验/1-2/roberta-loss_gap_sst2_0.0_5e-08_3404.csv','实验/1-2/gpt2-loss_gap_sst2_0.0_5e-08_3404.csv','实验/1-2/t5-loss_gap_sst2_0.0_5e-08_3404.csv']
    # show_loss_gap_4(file)
    #1-4
    file2 = [["实验/1-4/scores_bert-base-uncased_qnli_3404_5e-08_32.csv","实验/1-4/scores_roberta-base_qnli_3404_5e-08_32.csv",
              "实验/1-4/scores_gpt2_qnli_3404_5e-08_32.csv","实验/1-4/scores_t5_qnli_3404_5e-08_32.csv"],
             ["实验/1-4/scores_bert-base-uncased_mnli_3404_5e-08_32.csv","实验/1-4/scores_roberta-base_mnli_3404_5e-08_32.csv",
              "实验/1-4/scores_gpt2_mnli_3404_5e-08_32.csv","实验/1-4/scores_t5_mnli_3404_5e-08_32.csv"]]
    # show_s1(file2)
    #1-7
    file="实验/1-7/loss_gap_mrpc_0.0_5e-08_3404.csv"
    # show_loss_gap(file)
    file = ["实验/1-8/loss_gap_sst2_0.0_5e-08_3404.csv","实验/1-8/loss_gap_stsb_0.0_5e-08_3404.csv"]
    show_lossgap_1_8(file)
    #2-1
    file3 = [["实验/2-1/pooler_gap_1_mrpc_5e-08_42.csv","实验/2-1/pooler_gap_0_mrpc_5e-08_42.csv"],
             ["实验/2-1/pooler_gap_0_sst2_5e-08_3404.csv","实验/2-1/pooler_gap_1_sst2_5e-08_3404.csv"]]
    # show_loss_gaps_2(file3)

    #2-2
    file4 = [[["实验/2-2/mnli/u2-eval-acc.csv", "实验/2-2/mnli/d2-eval-acc.csv"],
             ["实验/2-2/mnli/u2-eval-loss.csv", "实验/2-2/mnli/d2-eval-loss.csv"],
             ["实验/2-2/mnli/u2-train-loss.csv", "实验/2-2/mnli/d2-train-loss.csv"]],
             [["实验/2-2/qqp/u2-eval-acc.csv", "实验/2-2/qqp/d2-eval-acc.csv"],
              ["实验/2-2/qqp/u2-eval-loss.csv", "实验/2-2/qqp/d2-eval-loss.csv"],
              ["实验/2-2/qqp/u2-train-loss.csv", "实验/2-2/qqp/d2-train-loss.csv"]]
             ]

    #2-3
    file = ['实验/2-3/pooler_gap_0_2e-08_mrpc.csv','实验/2-3/pooler_gap_0_5e-08_mrpc.csv','实验/2-3/pooler_gap_0_1e-07_mrpc.csv',
            '实验/2-3/pooler_gap_1_2e-08_mrpc.csv','实验/2-3/pooler_gap_1_5e-08_mrpc.csv','实验/2-3/pooler_gap_1_1e-07_mrpc.csv']
    # show_pooler_2_3(file)
    #2-4
    file = ['实验/2-4/pooler_gap_1_5e-08_mrpc.csv', '实验/2-4/pooler_gap_0_5e-08_mrpc.csv',
            '实验/2-4/pooler_gap_1_5e-08_mrpc.csv', '实验/2-4/pooler_gap_1_5e-08_mrpc.csv']
    # show_pooler_2_4(file)
    # 2-5
    file = ['实验/2-5/pooler_gap_0_5e-08_mrpc.csv', '实验/2-5/pooler_gap_0_6e-08_mrpc.csv',
            '实验/2-5/pooler_gap_0_2e-07_mrpc.csv']
    # show_pooler_2_5(file)
    la="实验/2-4/label_1_mrpc.csv"
    # show_loss_gap_(file[0],la)
    # show_loss_acc(file4)
    #损失颗粒发现
    file5=['实验/损失颗粒发现/loss_gap_0.005.csv','实验/损失颗粒发现/loss_gap_0.0005.csv','实验/损失颗粒发现/loss_gap_5e-05.csv','实验/损失颗粒发现/loss_gap_5e-06.csv',
           '实验/损失颗粒发现/loss_gap_5e-07.csv','实验/损失颗粒发现/loss_gap_5e-08.csv','实验/损失颗粒发现/loss_gap_3e-08_32.csv','实验/损失颗粒发现/loss_gap_2e-08_32.csv']
    # show_loss_gap_8(file5)
    #损失颗粒和输出颗粒
    file6 = [["实验/损失颗粒和输出颗粒/loss_gap_5e-08.csv"],
            ["实验/损失颗粒和输出颗粒/pooler_gap_0_bert_mrpc_3404_5e-08_32.csv"]]
    # show_lossgap_poolergap(file6)
    file = [["实验/1-4/scores_bert-base-uncased_mrpc_3404_5e-08_32.csv"],["实验/1-4/scores_bert-base-uncased_sst2_3404_5e-08_32.csv"]]
    file2 = (["score/scores_sst2_3404_up_0.0_1_0.0_5e-08.csv","score/scores_sst2_3404_up_0.0_2_0.0_5e-08.csv"])
    # show_loss_gap_(file3)

    file3 = [["32/pooler_gap_1_mnli_8e-08_3404.csv", "32/pooler_gap_2_mnli_8e-08_3404.csv"],
             ["32/pooler_gap_1_mrpc_5e-07_3404.csv", "32/pooler_gap_2_mrpc_5e-07_3404.csv"]]#,"实验/loss_gap_cola_5e-08_3404.csv" "实验/loss_gap_rte_5e-08_3404.csv",,"实验/loss_gap_rte_5e-08_3404.csv"
    # show_vectors(files)
    # show_loss_gaps_2(file3)
    loss = "32/pooler_gap_1_mrpc_5e-07_3404.csv"
    # show_loss_gap_(loss)
    print(3.5762786865234375e-07/5.960464577539063e-08)
    a=(7.43%2.1)/2.1
    b=(7.43/2.1)%1
    print(a,b)