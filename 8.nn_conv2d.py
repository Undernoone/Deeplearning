import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='/dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)

data_loader = DataLoader(dataset,batch_size=64)

class Coder729(nn.Module):
    def __init__(self):
        super(Coder729, self).__init__()
        self.conv1 = Conv2d(3, 6, 5,stride=1,padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

coder729 = Coder729()
print(coder729)

writer = SummaryWriter("8_nn_conv2d_Log")
step = 0
for data in data_loader:
    imgs,targets = data
    outputs = coder729(imgs)
    writer.add_images('input', imgs, step)
    # output = torch.reshape(outputs,(-1,3,30,30))
    writer.add_images("output",outputs,step)

writer.close()
