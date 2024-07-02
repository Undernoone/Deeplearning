import cv2
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("4_Transforms_Using_Logs")

img = Image.open('image/preview.jpg')
print(type(img))

img2 = "image/coder729.jpg"
img_cv = cv2.imread(img2)
print(type(img_cv))

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
print(type(img_tensor))
writer.add_image("ToTensor", img_tensor, 0)
img_tensor_cv = trans_totensor(img_cv)
print(type(img_tensor_cv))

# Normalize
trans_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize", img_norm)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
print(img_resize.size)
writer.add_image("Resize", img_resize)

# Compose
trans_resize_2 = transforms.Resize((256, 256))
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Compose", img_resize_2)

# RandomCrop
trans_randomcrop = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_randomcrop, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()
