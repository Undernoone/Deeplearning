import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_date = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(),)

# shuffle=False 保证数据集顺序不变
test_loader = DataLoader(test_date, batch_size=4, shuffle=False,num_workers=0,drop_last=False)

# 数据集第一章图片及标签
img,target = test_date[0]
print(img.shape)
print(target)

writer = SummaryWriter('6_DataLoader_Logs')
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs,target = data
        # print(img.shape)
        # print(target)
        writer.add_images("Epoch:()".format(epoch),imgs,step)
        step += 1

writer.close()
