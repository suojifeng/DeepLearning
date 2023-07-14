import torch
from torch import nn

# def init_xavier(m):
#     if type(m) == nn.Linear:
#         nn.init.xavier_uniform_(m.weight)
# def init_42(m):
#     if type(m) == nn.Linear:
#         nn.init.constant_(m.weight, 42)
# def init_constant(m):
#     if type(m) == nn.Linear:
#         nn.init.constant_(m.weight, 1)
#         nn.init.zeros_(m.bias)
# def init_normal(m):
#     if type(m) == nn.Linear:
#         nn.init.normal_(m.weight, mean=0, std=0.01)
#         nn.init.zeros_(m.bias)
#
# net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
# X = torch.rand(size=(2, 4))
# net(X)
# print(net[2].state_dict())
# net.apply(init_normal)
# net[0].weight.data[0], net[0].bias.data[0]
print(torch.cuda.device_count())