import torch
torch.manual_seed(42)
import torch
def func(w,x):
    y=torch.matmul(w,x)
    return torch.sum(y ** 2)
def loss_func(w, x,v):
    loss=func(w,x)
    loss.backward(create_graph=False)
    g=w.grad
    w.grad = None
    # g = torch.autograd.grad(func(w, x), w, create_graph=True)[0]
    print(f'g:{g}')
    gv = torch.dot(g.ravel(), v.ravel())
    return gv


# power iteration 方法计算 Hessian 最大特征值
def power_iteration(w,x, max_iter=100, eps=1e-6):


    # 初始化随机单位向量
    v = torch.randn_like(w)
    v /= torch.norm(v)
    v_old = v.clone()  # Initialize v_old

    for i in range(max_iter):
        # 计算 Hv
        v.requires_grad = True
        loss=loss_func(w, x,v)
        loss.backward()
        Hv=w.grad
        # Hv =  torch.autograd.grad(loss_func(w, x,v), w)[0]
        w.grad = None
        print(f"Gradient of gv with respect to w: {Hv}")

        # 更新 v
        v = Hv / torch.norm(Hv)

        # 检查收敛条件
        if torch.abs(v - v_old).max() < eps:
            break
        v_old = v.clone()

    # 计算最大特征值
    max_eigenvalue = torch.dot(v.ravel(), Hv.ravel())
    return max_eigenvalue

def gradmask():
    mask=torch.tensor([
     [1,0,1],[0,0,1],[1,1,0]
    ])
    W = torch.randn(3, 3, requires_grad=True)
    X = torch.randn(3, requires_grad=True)
    w=W*mask
    y = torch.matmul(w, X)
    loss= torch.sum(y ** 2)
    loss.backward()
    print(W.grad)


def gradmask1():
    mask = torch.tensor([
        [1, 0, 1],
        [0, 0, 1],
        [1, 1, 0]
    ], dtype=torch.float32,requires_grad=True)

    W = torch.randn(3, 3, requires_grad=True)
    X = torch.randn(3, requires_grad=True)
    # 使用掩码进行前向计算
    w = W
    # w.retain_grad()
    y = torch.matmul(w, X)

    # 计算损失
    loss = torch.sum(y**2)

    # 计算梯度
    loss.backward()

    # 打印原始梯度
    print("原始梯度：")
    print(w.grad)
# 获取需要计算的参数和向量
W = torch.randn(3,3, requires_grad=True)

X = torch.randn(3, requires_grad=True)
# 计算 Hessian 最大特征值
# max_eigenvalue = power_iteration(W,X)
# print(f"Hessian matrix's maximum eigenvalue: {max_eigenvalue}")
# g=torch.tensor([[1.0, 2.0], [3.0, 4.0]])
# v=torch.tensor([[1.0, 2.0], [3.0, 4.0]])
# gv = torch.dot(g.ravel(), v.ravel())
# print(gv)
gradmask1()