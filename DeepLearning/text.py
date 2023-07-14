import torch

# 构建输入张量
x = torch.tensor(5.0, requires_grad=True)

# 定义函数
def f(x):
    y = x ** 2 + 2 * x + 1
    return y

# 前向传播
y = f(x)
print("前向传播结果: ", y)

# 反向传播
y.backward()

# 输出梯度
print("梯度: ", x.grad)