import torch
import numpy

from assignment2 import MultitaskNetwork, multitask_training, mlb_position_player_salary

print("=== Test MultitaskNetwork ===")
net = MultitaskNetwork()
x = torch.randn(4, 3)
ya, yb = net(x)
print("ya shape:", ya.shape, "yb shape:", yb.shape)
print("ya row sums:", ya.sum(dim=1))
print("yb row sums:", yb.sum(dim=1))

print("\n=== Test multitask_training ===")
trained_net = multitask_training("multitask_data.csv")
x2 = torch.randn(2, 3)
pa, pb = trained_net(x2)
print("trained outputs:", pa.shape, pb.shape)

print("\n=== Test mlb_position_player_salary ===")
model, perf = mlb_position_player_salary("baseball.txt")
print("is module:", isinstance(model, torch.nn.Module))
print("val perf:", perf)
