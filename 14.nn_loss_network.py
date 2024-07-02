import torch
import torchvision
from torch import nn, conv2d
from torch.nn import Conv2d, MaxPool2d, Sequential, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='/dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)
data_loader = DataLoader(dataset,batch_size=1)


class Coder729(nn.Module):
    def __init__(self, ):
        super(Coder729, self).__init__()
        self.model1 = Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10),
        )

    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
coder729 = Coder729()
for data in data_loader:
    imgs,targets = data
    outputs = coder729(imgs)
    result_loss = loss(outputs,targets)
    result_loss.backward()
    print("result_loss")