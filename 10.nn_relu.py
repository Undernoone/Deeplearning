import torch
import torchvision
from torch import nn
from torch.nn import ReLU
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# 非线性变换
# input = torch.tensor([[1,-0.5],
#                       [-1,3]])
#
# output = torch.reshape(input, (-1,1,2,2))
# print(output.shape)

dataset = torchvision.datasets.CIFAR10(root='D:\\Deep_learning\\dataset',train=False,transform=torchvision.transforms.ToTensor(),download=False)

data_loader = DataLoader(dataset,batch_size=64)

class Coder729(nn.Module):
    def __init__(self):
        super(Coder729, self).__init__()
        self.relu = ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output

# coder729 = Coder729()
# output = coder729(input)
# print(output)

coder729 = Coder729()

writer = SummaryWriter("10_nn_relu_Logs")
step = 0
for data in data_loader:
    imgs,targets = data
    writer.add_images("input",imgs,step)
    output = coder729(imgs)
    writer.add_images("output",output,step)
    step += 1

writer.close()




















