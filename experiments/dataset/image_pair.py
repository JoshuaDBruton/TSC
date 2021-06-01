from torch.utils.data import Dataset
import numpy as np
import os
import torch
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
# from google_drive_downloader import GoogleDriveDownloader as gdd


class ImageSegmentation(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None, concat_coords=False):
        """
        DataSet for representing pairs of query and reference images (saved as .jpg/.png files) with full pixel-level
        labels
        Two folders, inputs, for the query images, and targets, for the reference images, are required to be present in
        the top-level root directory
        :param root_dir: the root directory, contain folders train and test, each containing train and validation folders
        :param transform: A transform to be applied to the inputs
        :param target_transform: A transform to be applied to the targets/references
        :param concat_coords: Whether or not to concatenate (x, y) coord on to each pixel
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.concat_coords = concat_coords

        self.inputs_dir = os.path.join(self.root_dir, "inputs")
        self.targets_dir = os.path.join(self.root_dir, "targets")

        # gdd.download_file_from_google_drive(file_id=file_id, dest_path=self.root_dir+".zip", unzip=True)

        self.n = len(os.listdir(self.inputs_dir))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = np.array(plt.imread(os.path.join(self.inputs_dir, str(idx)+".jpg")))
        # y = np.array(plt.imread(os.path.join(self.targets_dir, str(idx)+".png")))
        y = np.load(os.path.join(self.targets_dir, str(idx)+".npy"))

        # if len(x.shape)<3:
        #     x = transforms.ToPILImage()(x)
        #     x = transforms.Grayscale(3)(x)
        #     x = transforms.ToTensor()(x)

        assert (len(x.shape) == 3), "Input x does not have 3 dimensions, shape is {}".format(x.shape)
        assert (x.shape[2] == 3), "Input x is not RGB, does not have 3 channels; x has {} channels".format(x.shape[2])
        assert (len(y.shape) == 2), "Shape of y is {}, not 2-dim grayscale".format(y.shape)
        assert (x.dtype == np.uint8), "Type of x is not uint8, it is {}".format(x.dtype)
        assert (y.dtype == np.uint8), "Type of y is not uint8, it is {}".format(y.dtype)

        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            y = torch.as_tensor(np.array(y), dtype=torch.int64).unsqueeze(0)
            y = self.target_transform(y)

        # Concatenating normalised (x,y) coordinate to each pixel, so network requires 2 extra channels (RGBXY)
        if self.concat_coords:
            gx, gy = torch.meshgrid(torch.arange(0, x.shape[1]), torch.arange(0, x.shape[2]))
            gx = gx/torch.max(gx)
            gy = gy/torch.max(gy)
            gx = gx.unsqueeze(0)
            gy = gy.unsqueeze(0)
            x = torch.cat((x, gx, gy), 0)

        sample = (x, y)

        return sample