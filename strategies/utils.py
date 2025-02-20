"""Auxiliary heuristics for choosing parameters

These functions are implemented as purely numpy functions for ease
of debugging and interpretation. They are then plugged into the
rest of the framework pipeline
"""

import numpy as np
import torch



def fraction_threshold(tensor, fraction):
    """Compute threshold quantile for a given scoring function

    Given a tensor and a fraction of parameters to keep,
    computes the quantile so that only the specified fraction
    are larger than said threshold after applying a given scoring
    function. By default, magnitude pruning is applied so absolute value
    is used.

    Arguments:
        tensor {numpy.ndarray} -- Tensor to compute threshold for
        fraction {float} -- Fraction of parameters to keep

    Returns:
        float -- Threshold
    """
    threshold,_= torch.topk(tensor, int((fraction)*len(tensor)))

    return threshold[-1]




def threshold_mask(tensor, threshold):
    """Given a fraction or threshold, compute binary mask

    Arguments:
        tensor {numpy.ndarray} -- Array to compute the mask for

    Keyword Arguments:
        threshold {float} -- Absolute threshold for dropping params

    Returns:
        np.ndarray -- Binary mask
    """
    assert isinstance(tensor, torch.Tensor)
    idx = tensor < threshold
    mask = torch.ones_like(tensor,device=torch.device('cuda:0'))
    mask[idx] = 0
    return mask


def fraction_mask(tensor, fraction):
    assert isinstance(tensor, np.ndarray)
    threshold = fraction_threshold(tensor, fraction)
    return threshold_mask(tensor, threshold)


def flatten_importances(importances):
    return torch.cat([
        importance.flatten()
        for _, importance in importances.items()
    ],dim=0)


def map_importances(fn, importances):
    return {module: fn(importance)
            for module, importance in importances.items()}


def importance_masks(importances, threshold):
    return map_importances(lambda imp: threshold_mask(imp, threshold), importances)
    # return {module:
    #         {param: threshold_mask(importance, threshold)
    #             for param, importance in params.items()}
    #         for module, params  in importances.items()}


def norms_tensor(tensor, ord, matrix_mode=False):
    if matrix_mode:
        assert len(tensor.shape) > 2
        tensor_flattened = tensor.reshape(*tensor.shape[:2], -1)
    else:
        tensor_flattened = tensor.reshape(tensor.shape[0], -1)
    norms = []
    for w in tensor_flattened:
        norms.append(np.linalg.norm(w, ord))
    return np.array(norms)

#计算最大特征值
def power_iteration(w,g, max_iter=5, eps=1e-6):
    # 初始化随机单位向量
    v = torch.randn_like(g)
    v = v / torch.norm(v)
    v_old = v.clone()  # Initialize v_old
    # g_old = g.clone()
    for i in range(max_iter):
        # 计算 Hv
        # print(f'2g:{g_old}')
        gv = torch.dot(g.ravel(), v.ravel()) #torch==2.1.0 -q
        gv.backward(retain_graph=True)
        Hv=w.grad
        # Hv =  torch.autograd.grad(gv, w,retain_graph=True)[0]
        # print(f'2w:{w.grad}')
        w.grad=None
        # 更新 v
        v = Hv / torch.norm(Hv)
        # print(f'2v:{v}')
        # 检查收敛条件
        if torch.abs(v - v_old).max() < eps:
            break
        v_old = v.clone()
    # 计算最大特征值
    max_eigenvalue = torch.dot(v.ravel(), Hv.ravel()).item()
    return max_eigenvalue






