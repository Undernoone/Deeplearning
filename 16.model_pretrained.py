import torchvision
from torchvision.models import VGG16_Weights

# 加载没有预训练权重的 VGG16 模型
vgg16_false = torchvision.models.vgg16(weights=None)

# 加载有预训练权重的 VGG16 模型
vgg16_true = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

print('ok')
