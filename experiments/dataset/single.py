from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from torchvision import transforms


class SingleImage(Dataset):
    """
    DataSet for representing single images saved as any image readable by plt.imread
    Images should be at the top-level root directory
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.n = len(os.listdir(self.root_dir))
        self.list_of_image_names = sorted(os.listdir(self.root_dir))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = Image.open(os.path.join(self.root_dir, self.list_of_image_names[idx]))

        if self.transform:
            x = self.transform(x)

        return x
