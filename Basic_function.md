### Tensoroard：可视化工具

### SummaryWriter：训练日志写入

```python
writer = SummaryWriter("2_TensorBoard_Logs")  # 指定日志目录
```

参数：log_dir为日志目录名（输出结果的文件夹名），启动TensorBoard的指令: tensorboard--logdir=“目录名”

### add_image：将图像数据添加到TensorBoard的日志目录中

```python
def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats="CHW")
```

```python
writer.add_image("test", img_array, 1, dataformats="HWC")
```

参数：tag为图片名称；img_tensor为数据类型，格式只能为tensor（pytorch）或array（numpy）；global_step为训练次数；walltime为现实训练时间；dateformats为图像的数据格式

### add_scalar：记录数据，常记录loss值

```python
def add_scalar(self, tag,scalar_value, global_step=None, walltime=None, new_style=False,  double_precision=False,):
```

参数：tag为图片名称；value为数值，step为步骤

### Transforms：可视化工具

### ToTensor：将PIL或numpy格式图像转换为tensor型格式（opencv	创造的图形都是numpy格式）

![img](https://coder729.oss-cn-beijing.aliyuncs.com/Typora/wps7.jpg)

### Normalize：将数据归一化（归一化可提高训练速度）

```python
class Normalize(torch.nn.Module):
#   Normalize a tensor image with mean and standard deviation.This transform does not support PIL Image.
```

参数：需要传入torch格式，不支持PIL

### Resize：重设图片大小

### Compose：将多个操作组合成一个单一操作，传入数据需是列表

### RandomCrop：随机裁剪

### Torchvision_Dateset：Pytorch提供的数据集

```python
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)
```

参数：root为下载路径；train为判断是否为训练数据集参数设置；transform为图像需要进行的变换操作，一般使用compose把所需的transforms结合起来。

### DataLoader：可迭代的数据装载器

```python
class DataLoader(Generic[T_co]):
    dataset: Dataset[T_co]
    batch_size: Optional[int]
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float
    sampler: Union[Sampler, Iterable]
    pin_memory_device: str
    prefetch_factor: Optional[int]
    _iterator : Optional['_BaseDataLoaderIter']
    __initialized = False
```

```python
test_loader = DataLoader(test_date, batch_size=4, shuffle=False,num_workers=0,drop_last=False)
```

参数：dateset为引入的数据集，batchsize为批次数，shuffle为是否洗牌，num_workers为是否多进程，droplast为训练数据不足size时候是否舍弃最后一批数据

## conv2d：二维卷积

```python
def __init__(
    self,in_channels: int,out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1,
    padding: Union[str, _size_2_t] = 0,
    dilation: _size_2_t = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: str = 'zeros',  # TODO: refine this type
    device=None,
    dtype=None
)
```

```python
self.conv1 = Conv2d(3, 6, 5,stride=1,padding=0)
```

参数：in_channel为输入图片通道数；out_channel为输出图片通道数；kernel_size为卷积核的大小；stride为步长；padding为边距

输出图片的长与宽的计算：

<img src="https://coder729.oss-cn-beijing.aliyuncs.com/Typora/wps8.jpg" alt="img" style="zoom:150%;" />

### 输入、输出的通道数：彩色图像有高，宽和RGB3个通道。

假设彩色图像的高和宽分别是hhh和www（像素），那么它可以表示为一个3 × h × w 的多维数组。将大小为三的这一维成为通道数。

### stride 和 padding 的可视化



### 最大池化的作用和目的：最大限度的保留图片特征，同时减少数据量。加速训练速度*

 

### CrossEntropyLoss：交叉熵损失函数