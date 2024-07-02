import torchvision
import torch

vgg16 = torchvision.models.vgg16(weights=False)

torch.save(vgg16, 'vgg16_method1.pth')
