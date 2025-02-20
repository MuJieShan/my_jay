# -- coding: utf-8 --
from torch.utils.data.dataloader import DataLoader
import numpy as np
from transformers.data.data_collator import DataCollatorWithPadding
import torch
from transformers import BertTokenizer,AutoTokenizer
from torch.utils.data import SubsetRandomSampler
from datasets import Dataset

class GLUEPruner():
    def __init__(self, dataset,ratio = 0.5,pruneFlag="up"):
        self.dataset = dataset
        self.keep_ratio =1-ratio if ratio<=1 else (100-ratio)/100
        self.index=np.arange(len(self.dataset))
        self.device=torch.device('cuda:0')
        self.scores = torch.zeros([len(self.dataset)]).cpu()
        self.cur_batch_index = None
        self.iteration=0
        self.cur_index=None
        self.num_pruned_samples=0
        self.state=pruneFlag
    def update(self, values,index):
        self.cur_batch_index=index
        # s=torch.abs(values.to(dtype=self.scores.dtype).cpu())
        s=values.to(dtype=self.scores.dtype).cpu()
        self.scores[self.cur_batch_index.cpu()]=s
        # self.scores[self.cur_batch_index.cpu()] = s

    def random_prune(self):
        num = int(self.keep_ratio * len(self.index))
        remain_indices = np.random.choice(self.index, size=num, replace=False)
        print('remain_samples', len(remain_indices))
        np.random.shuffle(remain_indices)
        self.cur_index = remain_indices
        self.iteration += 1

    def prune(self):
        if self.iteration==0:
            remain_indices=np.arange(len(self.dataset))
            np.random.shuffle(remain_indices)
            self.cur_index = remain_indices
            self.iteration += 1
            return
        def fraction_threshold(tensor, fraction):
            if (self.state == "up"):
                threshold, _ = torch.topk(tensor, int((fraction) * len(tensor)))
            else:
                threshold, _ = torch.topk(-tensor, int((fraction) * len(tensor)))
                threshold = -threshold
            return threshold[-1]

        def threshold_mask(tensor, threshold):
            assert isinstance(tensor, torch.Tensor)
            if (self.state == "up"):
                idx = tensor < threshold
            else:
                idx = tensor > threshold
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
        print(type(self.dataset))
        # index = [2989, 2508, 739, 2155, 2191, 518, 3502, 2305, 223, 2767, 732, 2884, 190, 3650, 3265, 1789, 2681, 1462, 1033, 852, 1084, 1732, 540, 3572, 3602, 2864, 1451, 1203, 743, 3085, 1227, 1385, 1924, 2051, 2217, 1150, 1239, 2356, 2255, 2081, 2049, 1493, 278, 2455, 2507, 3335, 2518, 765, 797, 2992, 2856, 1884, 3442, 3216, 1071, 11, 2634, 503, 3331, 2586, 1664, 363, 3283, 894, 835, 3076, 314, 1375, 3038, 949, 2994, 1249, 1582, 906, 3214, 579, 2427, 1713, 2268, 1533, 1080, 2379, 3553, 2917, 2205, 1752, 2063, 92, 2330, 1908, 782, 1244, 1294, 1405, 624, 3070, 1567, 2543, 426, 594, 3080, 3256, 919, 2122, 2132, 1458, 2572, 5, 3077, 1796, 868, 879, 2844, 1318, 834, 1314, 567, 3200, 462, 3379, 2243, 2716, 1888, 3562, 3264, 1896, 467, 791, 1831, 1914, 2301, 556, 1694, 1521, 2863, 2002, 81, 3221, 1568, 2298, 2936, 2394, 2552, 491, 2673, 286, 2781, 756, 1620, 109, 725, 2622, 2698, 3412, 3305, 3355, 1867, 2249, 612, 1667, 3058, 998, 1292, 2748, 837, 2973, 219, 570, 2405, 650, 986, 3096, 625, 961, 2248, 1826, 3663, 3082, 2536, 29, 2787, 3455, 3241, 3191, 1665, 2076, 381, 412, 3036, 1592, 2945, 3262, 3570, 1659, 2840, 2367, 2755, 2986, 3513, 187, 2570, 1516, 1069, 1050, 2354, 1646, 989, 3005, 2239, 841, 1513, 1562, 1800, 686, 3614, 101, 3604, 1070, 2488, 2467, 1554, 2491, 2364, 486, 1353, 2434, 1860, 621, 2783, 1384, 2664, 1173, 2066, 1154, 1902, 3366, 49, 1444, 1625, 1980, 2679, 2397, 3436, 1283, 1161, 1912, 2941, 1666, 3458, 2720, 511, 772, 2760, 1852, 205, 2534, 887, 1828, 1015, 925, 847, 3346, 911, 1925, 2523, 978, 3134, 2276, 611, 2583, 1189, 3529, 2416, 2121, 2153, 3149, 2097, 1450, 2998, 7, 632, 599, 2214, 979, 2852, 3633, 1971, 162, 3240, 1441, 2927, 1412, 3482, 2082, 3088, 2566, 356, 1718, 2618, 224, 3348, 1814, 646, 2612, 398, 432, 3557, 738, 3514, 3097, 3278, 3171, 2135, 718, 2154, 1373, 1124, 719, 1096, 1101, 1687, 2678, 3370, 1639, 813, 2166, 2162, 2012, 3293, 2378, 2294, 1640, 1396, 1297, 1514, 2125, 290, 2466, 1635, 3365, 3071, 1879, 1316, 1237, 1600, 132, 2172, 1816, 255, 1218, 1029, 189, 2297, 3004, 2916, 368, 809, 669, 1528, 1322, 3208, 1731, 416, 1286, 2693, 3464, 643, 358, 729, 2924, 2811, 746, 857, 2145, 1118, 927, 3474, 1560, 1830, 3218, 1871, 1854, 3369, 3115, 793, 100, 2725, 548, 2370, 3242, 3129, 438, 3306, 1014, 74, 1296, 207, 3501, 1845, 1192, 229, 1282, 2838, 3202, 895, 2961, 408, 3437, 1449, 2185, 1471, 2167, 3170, 613, 3588, 1920, 1135, 1081, 451, 480, 3074, 3253, 2380, 2653, 642, 3428, 3176, 1039, 1303, 2697, 3635, 3517, 21, 3034, 3261, 8, 487, 1086, 2078, 1550, 2890, 1915, 3411, 1990, 2599, 3569, 311, 1928, 1310, 2190, 1087, 199, 233, 2792, 3467, 127, 1003, 954, 2567, 2494, 676, 1397, 2273, 1079, 239, 1054, 1321, 1754, 3478, 1304, 1077, 19, 2215, 3209, 372, 993, 312, 3228, 1436, 370, 1657, 2614, 391, 933, 1695, 982, 930, 3353, 3640, 2283, 2072, 658, 3427, 1245, 472, 38, 14, 1165, 1654, 1119, 54, 2210, 923, 2613, 1899, 1094, 3376, 2025, 251, 1068, 2798, 1007, 774, 1955, 2101, 1062, 1306, 3016, 2820, 3512, 569, 1733, 2202, 2413, 1978, 536, 315, 2264, 453, 2563, 1515, 2728, 619, 1839, 3296, 3068, 3445, 957, 3558, 3389, 2456, 240, 2218, 1989, 3390, 997, 57, 2042, 2668, 1457, 1842, 527, 3328, 2390, 380, 2236, 247, 2898, 2817, 2272, 2702, 802, 696, 600, 1392, 1048, 1293, 1598, 2946, 667, 590, 2331, 476, 227, 1558, 1139, 1962, 26, 3565, 2576, 2978, 1575, 196, 683, 2971, 1486, 2601, 284, 102, 2860, 1615, 1024, 3503, 1058, 610, 976, 2226, 2015, 3300, 1489, 2152, 166, 1207, 3067, 1372, 53, 1335, 220, 635, 321, 1200, 2519, 3405, 2738, 1209, 1117, 3420, 633, 545, 3539, 3, 1269, 1298, 413, 3055, 1519, 2083, 790, 2278, 500, 2200, 2183, 731, 1798, 3116, 1415, 1287, 877, 209, 292, 904, 3177, 3537, 3627, 2407, 447, 2579, 221, 3205, 838, 2785, 2256, 378, 2424, 3623, 2384, 3594, 2966, 2976, 2605, 2549, 2117, 1076, 1760, 748, 2893, 2932, 546, 1063, 1043, 2475, 1356, 936, 1566, 2070, 2832, 58, 3148, 1420, 1824, 2350, 1114, 2412, 2551, 2520, 1064, 630, 2449, 2036, 2363, 327, 2223, 429, 216, 1650, 2515, 2441, 2375, 3595, 2888, 2216, 585, 1703, 1759, 3197, 2754, 2349, 2892, 2930, 1334, 2654, 2007, 2790, 3577, 1194, 2761, 3052, 157, 1317, 2499, 796, 2404, 2332, 443, 2306, 3315, 3015, 777, 672, 1185, 1, 1171, 299, 3551, 520, 3175, 3211, 1887, 9, 3493, 2965, 1975, 2393, 1049, 2027, 2230, 2750, 509, 1459, 2400, 856, 1427, 446, 2674, 260, 3291, 3314, 3492, 2265, 337, 382, 1110, 1947, 65, 966, 692, 82, 2757, 1409, 693, 2146, 110, 687, 2651, 1936, 1238, 460, 2169, 3489, 2198, 3113, 2335, 2319, 2464, 1751, 444, 2108, 839, 1869, 645, 522, 379, 2192, 2156, 1783, 2414, 318, 2609, 2124, 2858, 862, 421, 300, 2954, 1700, 2006, 331, 3204, 1363, 466, 1634, 3612, 2259, 1445, 1644, 2402, 1773, 2009, 2900, 1535, 1052, 3363, 2271, 1065, 845, 2023, 3578, 3203, 2435, 589, 1735, 18, 1226, 2157, 3274, 3387, 340, 16, 2901, 2091, 2069, 1853, 2204, 1961, 3193, 3402, 215, 344, 1933, 1755, 1808, 3510, 3081, 3576, 407, 1085, 2834, 1421, 851, 3002, 2535]
        # traindataset = self.dataset[index]
        traindataset = self.dataset[self.cur_index]
        # import random
        # unique_numbers = random.sample(index, 300)
        # # 打乱列表顺序
        # random.shuffle(unique_numbers)
        # traindataset = self.dataset[unique_numbers]
        # traindataset = traindataset[:300]
        train_dataset = Dataset.from_dict(traindataset)
        print(type(train_dataset))
        return train_dataset
    def get_largest_score(self,num):
        import heapq
        largest = heapq.nlargest(num, enumerate(self.scores), key=lambda x: x[1])
        largest_sorted = sorted(largest, key=lambda x: x[1], reverse=True)
        indices = [index for index, score in largest_sorted]
        dataset = self.dataset[indices]
        dataset = Dataset.from_dict(dataset)
        print(largest_sorted)
        return dataset
    def get_scores(self):
        scores = self.scores

        # 找到最大值和最小值
        max_score = max(scores)
        min_score = min(scores)

        # 计算差值
        range_score = max_score - min_score
        print(f'max_score{max_score}')
        print(f'min_score{min_score}')
        # 计算每个部分的范围
        part_range = range_score / 10

        # 初始化10个部分的列表
        parts = [[] for _ in range(10)]

        # 将分数分配到相应的部分
        for score in scores:
            # 计算当前分数应该属于哪个部分
            part_index = int((score - min_score) / part_range)
            # 防止因为浮点数精度问题导致的索引越界
            if part_index == 10:
                part_index = 9
            parts[part_index].append(score)

        # 打印每个范围的取值范围以及相应的个数
        for i in range(10):
            # 计算每个部分的取值范围
            lower_bound = min_score + i * part_range
            upper_bound = min_score + (i + 1) * part_range
            # 对于最后一个区间，上界应该是闭区间
            if i == 9:
                print(f"Range {i + 1}: [{lower_bound:.2f}, {max_score:.2f}] - Count: {len(parts[i])/len(scores)}")
            else:
                print(f"Range {i + 1}: [{lower_bound:.2f}, {upper_bound:.2f}) - Count: {len(parts[i])/len(scores)}")
def get_pruned_dataloader(config,dataset,sampler):
    model_checkpoint = config.model
    batch_size = config.batchsize
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_checkpoint)
    if config.pad_token_id == None:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.eos_token_id
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





