import torch
import torchvision
from torch import nn
from torch.nn import ReLU
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='/dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)
data_loader = DataLoader(dataset,batch_size=64)

class Coder729(nn.Module):
    def __init__(self):
        super(Coder729, self).__init__()
        self.linear1 = nn.Linear(196608, 10)

    def forward(self, input):
        output1 = self.linear1(input)
        return output1

coder729 = Coder729()
for data in data_loader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.reshape(imgs,(-1,3,32,32))
    print(output.shape)
    output = coder729(output)
    print(output.shape)
