# from SSFCN.metrics.jensenshannon import jensen_shannon
from SSFCN.metrics.mi_score import mi_score
from torchvision import transforms
from PIL import Image
from torch.functional import F
import operator
import torch
import os
from torch.nn import CrossEntropyLoss, NLLLoss2d


class HyperParameters:
    """
    Model and training hyper-parameters
    """
    img_size = (512, 512)
    bs = 4
    num_workers = 1
    lr = 1e-4
    epochs = 1
    unet_depth = 5
    unet_start_filters = 32
    # loss = mine_loss
    # loss = F.mse_loss
    # loss = F.kl_div
    # loss = jensen_shannon
    weights = torch.tensor([1.0, 67.9, 1.8, 9.4])
    loss = CrossEntropyLoss(weight=weights)
    # loss = NLLLoss2d()

    """
    Directory and dataset management
    """
    base = "/home/joshua/Desktop/Work/SSFCN"

    dataset_name = "shapes"

    data_dir = os.path.join(base, 'data')

    train_data_dir = os.path.join(data_dir, dataset_name, 'train')

    val_data_dir = os.path.join(data_dir, dataset_name, 'validation')

    test_data_dir = os.path.join(data_dir, 'wetlands.ai')

    downstream_data_dir = os.path.join(data_dir, 'downstream')

    results_dir = os.path.join(base, 'results')

    """
    Transforms to apply to each dataset, and option to invert normalisation for display
    """
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = train_transform

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Pad(30, padding_mode='reflect'),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    downstream_transform = transforms.Compose([
        transforms.Pad(30, padding_mode='reflect'),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )

    target_transform = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.ToTensor(),
        # transforms.Pad(30, padding_mode='reflect'),
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
    ])


def as_dict() -> dict:
    self = HyperParameters()
    return {'batch size': self.bs, 'epochs': self.epochs,}


def crop(img, bounding):
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]
