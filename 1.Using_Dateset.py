from torch.utils.data import Dataset
from PIL import Image
import os


class Mydate(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)

    def __len__(self):
        return len(self.img_path)


root_dir = "E:\\Deep_learning\\hymenoptera_data\\train"
label_dir = "ants"
