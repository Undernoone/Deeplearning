import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# 最大池化层
dataset = torchvision.datasets.CIFAR10(root='/dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)

data_loader = DataLoader(dataset,batch_size=64)

# input = torch.tensor([[1,2,0,3,1],
#                       [0,1,2,3,1],
#                       [1,2,1,0,0],
#                       [5,2,3,1,1],
#                       [2,1,0,1,1]],dtype=torch.float32)

# input = torch.reshape(input,(-1,1,5,5))
# print(input.shape)

class Coder729(nn.Module):
    def __init__(self):
        super(Coder729, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

coder729 = Coder729()
# output = coder729(input)
# print(output)

writer = SummaryWriter("9_nn_maxpool_Logs")
step = 0
for data in data_loader:
    imgs,targets = data
    writer.add_images("input",imgs,step)
    output = coder729(imgs)
    writer.add_images("output",output,step)
    step += 1

writer.close()