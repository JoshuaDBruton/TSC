from torch.utils.data import Dataset
import numpy as np
import os
import torch
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt


class ImageSegmentation(Dataset):
    """
    DataSet for representing pairs of query and reference images (saved as .npy files) with full pixel-level labels
    Two folders, inputs, for the query images, and targets, for the reference images, are required to be present in
    the top-level root directory
    """
    def __init__(self, root_dir="/home/joshua/Desktop/Work/SelfSuperNet/data/train", transform=None,
                 target_transform=None, concat_coords=False,):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.concat_coords = concat_coords

        self.inputs_dir = os.path.join(self.root_dir, "inputs")
        self.targets_dir = os.path.join(self.root_dir, "targets")
        self.n = len(os.listdir(self.inputs_dir))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = np.load(os.path.join(self.inputs_dir, str(idx)+".npy"))
        y = np.load(os.path.join(self.targets_dir, str(idx)+".npy"))

        # if len(x.shape)<3:
        #     x = transforms.ToPILImage()(x)
        #     x = transforms.Grayscale(3)(x)
        #     x = transforms.ToTensor()(x)

        if self.transform:
            x = self.transform(np.uint8(x))
            # x = torch.from_numpy(x)

        if self.target_transform:
            y = torch.as_tensor(np.array(y), dtype=torch.int64).unsqueeze(0)
            y = self.target_transform(y)

        if self.concat_coords:
            gx, gy = torch.meshgrid(torch.arange(0, x.shape[1]), torch.arange(0, x.shape[2]))
            gx = gx/torch.max(gx)
            gy = gy/torch.max(gy)
            gx = gx.unsqueeze(0)
            gy = gy.unsqueeze(0)
            x = torch.cat((x, gx, gy), 0)
            # x = x[3:5, :, :]

        sample = (x, y)

        return sample


def func(x):
    return x


def gen_x(data_size):
    return np.sign(np.random.normal(0., 1., [data_size, 1]))


def gen_y(x, var, data_size):
    return func(x)+np.random.normal(0., np.sqrt(var), [data_size, 1])


class SyntheticLinear(Dataset):
    def __init__(self, data_size=20000, var=0.2):
        self.data_size = data_size
        self.var = var

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x_sample = gen_x(self.data_size)
        y_sample = gen_y(x_sample, self.var, self.data_size)
        y_shuffle = np.random.permutation(y_sample)

        x_sample = Variable(torch.from_numpy(x_sample).type(torch.FloatTensor), requires_grad=True)
        y_sample = Variable(torch.from_numpy(y_sample).type(torch.FloatTensor), requires_grad=True)
        y_shuffle = Variable(torch.from_numpy(y_shuffle).type(torch.FloatTensor), requires_grad=True)

        return x_sample, y_sample, y_shuffle


class SyntheticLinearTest(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x = gen_x(262144)
        y = gen_y(x, 0.16, 262144)

        x = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=False)
        y = Variable(torch.from_numpy(y).type(torch.FloatTensor), requires_grad=False)

        return x, y
