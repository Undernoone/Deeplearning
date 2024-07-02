import torch
from torch import nn, conv2d
from torch.nn import Conv2d, MaxPool2d, Sequential, Flatten, Linear
from torch.utils.tensorboard import SummaryWriter


class Coder729(nn.Module):
    def __init__(self, ):
        super(Coder729, self).__init__()
        # self.conv1 = Conv2d(3,32,5,padding=2)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32,32,5,padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32,64,5,padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = nn.Flatten()
        # self.linear1 = nn.Linear(1024,64)
        # self.linear2 = nn.Linear(64,10)
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
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.model1(x)
        return x

coder729 = Coder729()
print(coder729)
input = torch.ones((64,3,32,32))
output = coder729(input)
print(output.shape)

writer = SummaryWriter("12_nn_Sequential_Logs")
writer.add_graph(coder729, input)
writer.close()