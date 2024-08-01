from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "E:\\Deep_learning\\hymenoptera_data\\train\\ants\\0013035.jpg"

img = Image.open(img_path)

# writer = SummaryWriter("3_Transforms_Logs")
# tensor_trans = transforms.ToTensor()
# tensor_img = tensor_trans(img)
# writer.add_image("Tenser_img", tensor_img)
#
# writer.close()

# 关于transforms如何使用和为什么使用tensor数据类型的注释：  
# 1. transforms的使用：transforms模块提供了许多预定义的转换方法，可以方便地对图片进行各种预处理操作，  
#    如缩放、裁剪、归一化等。这些转换方法通常用于数据加载和增强阶段，以便更好地训练神经网络。  
# 2. 为什么使用tensor数据类型：在深度学习中，神经网络通常使用张量（tensor）作为输入和输出。  
#    将图片转换为张量可以方便地与神经网络进行交互，并且可以利用PyTorch等深度学习框架提供的各种张量操作和优化方法。  
#    此外，张量也支持GPU加速，可以显著提高模型的训练速度。