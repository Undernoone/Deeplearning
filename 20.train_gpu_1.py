import torch
import torchvision
import time
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10(root='D:\\Deep_learning\\dataset',train=False,transform=torchvision.transforms.ToTensor(),download=False)
test_data = torchvision.datasets.CIFAR10(root='D:\\Deep_learning\\dataset',train=False,transform=torchvision.transforms.ToTensor(),download=False)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

train_loader = DataLoader(train_data,batch_size=64)
test_loader = DataLoader(test_data,batch_size=64)

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
            Conv2d(3,32,5,1,2),
            MaxPool2d(2),
            Conv2d(32,32,5,1,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,1,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10),
        )
    def forward(self, x):
        x = self.model1(x)
        return x

if __name__ == '__main__':
    model = Coder729()
    input = torch.randn(64,3,32,32)
    output = model(input)
    print(output.shape)

# 创建模型
model = Coder729()
model = model.cuda()
# 损失函数
loss_func = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 设置网络参数和训练测试次数
train_step = 0
test_step = 0

writer = SummaryWriter("19_train_log")
epochs = 10
start_time = time.time()
for epoch in range(epochs):
    print("第{}轮训练开始".format(epoch+1))
    for data in train_loader:
        imgs,targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = model(imgs)
        loss = loss_func(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_step += 1
        if train_step % 100 == 0:
            end_time = time.time()
            print(end_time-start_time)
            print('训练次数：{}，损失：{:.4f}'.format(train_step,loss))
            writer.add_scalar('train_loss', loss.item(), train_step)

    total_test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            imgs,targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = model(imgs)
            loss = loss_func(outputs, targets)
            total_test_loss += loss
    print("整体测试集上的损失：{:.4f}".format(total_test_loss))
    writer.add_scalar('test_loss', total_test_loss, epoch)
    test_step += 1

writer.close()