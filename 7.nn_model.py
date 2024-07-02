import torch
from torch import nn

class Coder729(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

coder729 = Coder729()
x = torch.tensor(1.0)
output = coder729(x)
print(output)