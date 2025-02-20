"""Magnitude based pruning

Implements pruning strategy that prunes as follows

For a given tensor, sort by absolute value and
keep only the _fraction_ with highest magnitudes
so that overall desired compression is achieved
"""

import numpy as np
import torch
import torch.nn.functional as F
from pruning import mask_module,Pruning
from .utils import (fraction_threshold,
                    fraction_mask,
                    map_importances,
                    flatten_importances,
                    importance_masks,
                    power_iteration
                    )



class pruner(Pruning):
    def __init__(self, model,compression=1,dataset_name=None,config=None,device=None,optimizer=None,log=None,lr_scheduler=None):
        super().__init__(model,compression=compression,dataset_name=dataset_name,config=config,device=device,log=log)
        self.g = dict()
        self.v = dict()
        self.m = dict()
        self.optimizer=optimizer
        self.lr_scheduler=lr_scheduler
        self.prune_epoch=None
        self.reg = {name : 0 for (name, module) in model.named_modules() if self.can_prune(module)}
        self.reg2 = {name : 0 for (name, module) in model.named_modules() if self.can_prune(module)}
        self.reg3 = 0
    def initmask(self,mask_file):
        masks = torch.load(mask_file)
        # print(masks.keys())
        for name, module in self.model.named_modules():
            if name in masks:
                mask = masks[name]
                self.masks[module] = mask
                # module.weight_mask.data.copy_(mask)
    def load_model_mask(self):
        for name, module in self.model.named_modules():
            if self.can_prune(module):
                mask = self.masks[module]
                module.weight_mask.data.copy_(mask)
    def layer_masks(self, module):
        alpha_1=self.config.alpha_1
        alpha_2 = self.config.alpha_2
        params = self.module_params(module)
        grads = self.module_param_gradients(module)
        if self.config.pruning_algo=='SEVEN':
            self.g[module] = torch.clone(torch.abs(grads)).detach().to(torch.device('cuda:0'))
            self.m[module] = alpha_1 * self.m[module] + (1 - alpha_1) * torch.abs(self.g[module])
            self.v[module] = alpha_2 * self.v[module] + (1 - alpha_2) * torch.pow(self.g[module], 2)
            m_hat = self.m[module] / (1 - (alpha_1 ** (self.prune_epoch + 1)))
            v_hat = (torch.sqrt(self.v[module] / (1 - (alpha_2 ** (self.prune_epoch + 1)))) + 1e-9)
            self.importances[module] += torch.abs(params*self.g[module]*m_hat/v_hat).to(torch.device('cuda:0'))
        elif self.config.pruning_algo=='PLATON':
            #PLATON
            self.g[module] = torch.clone(torch.abs(grads*params)).detach().to(torch.device('cuda:0'))
            self.m[module] = alpha_1 * self.m[module] + (1 - alpha_1) * torch.abs(self.g[module])
            self.v[module] = alpha_2 * self.v[module] + (1 - alpha_2) * torch.abs(self.g[module]-self.m[module])
            self.importances[module] = (self.m[module]*self.v[module]).to(torch.device('cuda:0'))

        elif 'movement' in self.config.pruning_algo:
            self.g[module] = torch.abs(grads)
            s = params*grads
            self.importances[module]= torch.abs(s).to(torch.device('cuda:0'))
    def update_reg(self):
        r3=[]
        for (name, module) in self.model.named_modules():
            if self.can_prune(module):
                # r=1/(module.weight.grad.abs().sum()+1000)#直接以梯度作为标准
                #实现以Hessian矩阵的最大特征值作为标准
                g=module.weight.grad
                # module.weight.grad=None
                # 获取梯度信息
                mask=self.masks[module]
                r=(torch.pow((1-mask)*g,2).sum()).item()
                # r=(torch.abs((1-mask)*g).sum()).item()
                # r=abs(power_iteration(module.weight,g))#求最大特征值
                # module.weight.grad=g
                self.reg[name] = r
                r2 = (torch.pow(mask * g, 2).sum()).item()
                # r2=(torch.abs(mask*g).sum()).item()
                self.reg2[name] = r2
                r3.append((torch.abs((1-mask)*module.weight).sum()).item()/torch.sum(1-mask).item())
        self.reg3 = sum(r3)/len(r3)

    def update_r(self):
        pruned=0
        remain=0
        for (name, module) in self.model.named_modules():
            if self.can_prune(module):
                g=module.weight.grad
                mask=self.masks[module]
                pruned += (torch.abs((1-mask)*g).sum()).item()
                remain += (torch.abs(mask*g).sum()).item()
        return pruned,remain
    def update_g(self):
        self.g=self.get_param_gradients()
    def freeze_pruned_grad(self):
        for name, module in self.model.named_modules():
            if self.can_prune(module):
                if module.weight.grad is not None:
                    module.weight.grad *= self.masks[module]
                # module.weight.detach()[self.masks[module]==0].requires_grad = False
    def freeze_pruned_weight(self):
        for name, module in self.model.named_modules():
            if self.can_prune(module):
                if module.weight is not None:
                    module.weight *= self.masks[module]
    def model_masks(self,iter_num=None):
        """Compute masks using the said strategy for every module
        This is a straight forward implementation that supports
        strategies that prune each module independently
        """
        if iter_num==0:
            num=0
            for module in self._prunable_modules():
                self.importances[module] = torch.zeros_like(module.weight.data,device=torch.device('cuda:0'))
                self.m[module] = torch.zeros_like(module.weight.data,device=torch.device('cuda:0'))
                self.v[module] = torch.zeros_like(module.weight.data,device=torch.device('cuda:0'))
                self.g[module] = torch.zeros_like(module.weight.data,device=torch.device('cuda:0'))
                self.masks[module]=torch.ones_like(module.weight.data,device=torch.device('cuda:0'))
                num+=1
            print('prunable layer num:',num)
            total_numel=0
            for _,i in self.m.items():
                total_numel+=i.numel()

        self.prune_epoch=iter_num
        remain_ratio = (self.keep_ratio) +(1-self.keep_ratio)*((1-(iter_num + 1) / self.config.iter_num)**(3))

        for module in self._prunable_modules():
            self.layer_masks(module)
        del self._param_gradients
        importances = self.importances
        if self.masks is not None:
            for m,mask in self.masks.items():
                importances[m][mask==0]=0
        flat_importances = flatten_importances(importances)

        threshold = fraction_threshold(flat_importances, remain_ratio)
        self.masks = importance_masks(importances, threshold)
        remain=0
        total=0
        for _, mask in self.masks.items():
            remain+=mask.sum().item()
            total+=mask.numel()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        mask_module(self.model, self.masks)
        return self.masks
