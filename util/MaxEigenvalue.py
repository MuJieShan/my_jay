import torch

torch.manual_seed(42)

def func(w, x):
    y = torch.matmul(w, x)
    return torch.sum(y ** 2)


def loss_func(g, v):
    gv = torch.dot(g.ravel(), v.ravel())
    return gv
# power iteration 方法计算 Hessian 最大特征值
def power_iteration(w,g, max_iter=100, eps=1e-6):
    # 初始化随机单位向量
    v = torch.randn_like(g)
    v = v / torch.norm(v)
    v_old = v.clone()  # Initialize v_old
    g_old = g.clone()
    for i in range(max_iter):
        # 计算 Hv
        gv = torch.dot(g_old.ravel(), v.ravel())
        # gv.backward(retain_graph=True)
        # Hv=w.grad
        Hv =  torch.autograd.grad(gv, w,retain_graph=True)[0]
        w.grad=None
        # 更新 v
        v = Hv / torch.norm(Hv)
        print(f'g:{g_old}')
        print(f'Hv:{Hv}')
        print(f'v:{v}')

        # 检查收敛条件
        if torch.abs(v - v_old).max() < eps:
            break
        v_old = v.clone()
    # 计算最大特征值
    max_eigenvalue = torch.dot(v.ravel(), Hv.ravel()).item()
    return max_eigenvalue


# 获取需要计算的参数和向量
w = torch.randn(4, 4, requires_grad=True)
x = torch.randn(4, requires_grad=True)

loss = func(w, x)
loss.backward(create_graph=True)
g = w.grad
w.grad = None
# 计算初始梯度
# g = torch.autograd.grad(func(w, x), w, create_graph=True)[0]#这个可以替换
print(f'g: {g}')

# 计算 Hessian 最大特征值
max_eigenvalue = power_iteration(w,g)
print(f"Hessian matrix's maximum eigenvalue: {max_eigenvalue}")
