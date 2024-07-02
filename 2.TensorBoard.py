from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("2_TensorBoard_Logs")  # 指定日志目录

image_path = "E:\\Deep_learning\\hymenoptera_data\\train\\ants\\0013035.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)  # add_image()函数要求输入的图片数据为numpy数组
print(type(img_array))
print(img_array.shape)

writer.add_image("test", img_array, 1, dataformats="HWC")

for i in range(100):
    writer.add_scalar("y=3x", 3 * i, i)  # 第一个参数是标签名，第二个参数是数值，第三个参数是步骤数

writer.close()
