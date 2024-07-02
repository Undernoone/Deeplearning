import torch
from torch import nn
from torch.nn import L1Loss, MSELoss

inputs = torch.tensor([1,2,3],dtype=torch.float)
targets = torch.tensor([1,2,6],dtype=torch.float)

inputs = torch.reshape(inputs, (1,1,1,3))
targets = torch.reshape(targets, (1,1,1,3))

loss = L1Loss(reduction='sum')
result = loss(inputs, targets)
print(result)

loss_mse = MSELoss(reduction='sum')
result_mse = loss_mse(inputs, targets)
print(result_mse)

x = torch.tensor([0.1,0.2,0.3],dtype=torch.float)
y = torch.tensor([1])
x = torch.reshape(x, (1,3))
loss_cross = nn. ()
result_cross = loss_cross(x, y)
print(result_cross)