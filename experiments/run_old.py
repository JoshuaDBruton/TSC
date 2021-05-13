import torch
from torch.utils.data import DataLoader
from SSFCN.model.TSCNet import TSCNet
from SSFCN.model.BNet import BNet
from SSFCN.model.mi_estimator import T, Mine
from pytorch_lightning import Trainer
from experiments.conf import HyperParameters as hp
from experiments.conf import as_dict
from experiments.dataset.image_pair import ImageSegmentation as DataSet
from experiments.dataset.pair import SyntheticLinear as MI_Data
from experiments.dataset.pair import SyntheticLinearTest as MI_test
from experiments.dataset.single import SingleImage
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.metrics.functional.classification import iou
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
import nibabel as nift
import torchvision.transforms as transforms
from PIL import Image

TRAIN_TRANSFORM = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

TARGET_TRANSFORM = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.ToTensor(),
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
])

def count_labels():
    all_data = DataSet(root_dir="/home/joshua/Desktop/data/ircad-dataset/validation", transform=TRAIN_TRANSFORM,
                       target_transform=TARGET_TRANSFORM)

    # all_data, _ = torch.utils.data.random_split(all_data, [10, len(all_data)-10])

    test_loader = DataLoader(all_data, shuffle=False, num_workers=hp.num_workers, batch_size=1,)

    bins = np.zeros(2)

    for (x, y) in tqdm(test_loader):
        counts, _ = np.histogram(y, bins=[0, 1, 2])
        bins += counts

    bins = bins/np.sum(bins)
    print()
    print(bins)
    plt.bar(x=[0, 1], height=bins, align='center')
    plt.xticks([0, 1], ['Background', 'Liver',])
    plt.show()

# Landcover train: [0.59 0.01 0.35 0.05]
# Landcover validation: [0.65 0.02 0.19 0.13]
# Landcover mean: [0.62 0.02 0.27 0.09]

# IILD train: [0.84 0.16]
# IILD validation: [0.87 0.13]
# IILD mean: [0.85 0.15]

# IRCAD train: [0.95 0.05]
# IRCAD validation: [0.96 0.04]
# IRCAD mean: [0.95 0.05]

def test_mi():
    data = DataSet(root_dir="/home/joshua/Desktop/Work/SSFCN/data/ircad/train", transform=hp.train_transform,
                   target_transform=hp.target_transform, concat_coords=True)
    val_data = DataSet(root_dir="/home/joshua/Desktop/Work/SSFCN/data/ircad/validation", transform=hp.train_transform,
                       target_transform=hp.target_transform, concat_coords=True)
    # test_data = DataSet(root_dir="/home/joshua/Desktop/Work/SSFCN/data/top_square/test", transform=hp.train_transform,
    #                     concat_coords=True)
    # split = int(len(data)*0.99)
    # train_data, val_data = torch.utils.data.random_split(data, [split, len(data)-split])
    # test_data, _ = torch.utils.data.random_split(test_data, [1, len(test_data)-1])
    # data, _ = torch.utils.data.random_split(data, [4, len(data)-4])
    # val_data, _ = torch.utils.data.random_split(val_data, [4, len(val_data)-4])

    train_loader = DataLoader(data, shuffle=True, num_workers=hp.num_workers, batch_size=4)
    val_loader = DataLoader(val_data, shuffle=False, num_workers=hp.num_workers, batch_size=4)
    # test_loader = DataLoader(test_data, shuffle=False, num_workers=hp.num_workers, batch_size=1)

    logger = WandbLogger(name="3-medical-3TSC-20",)

    # early_stopper = EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0.00,
    #     patience=5,
    #     verbose=False,
    #     mode='min'
    # )

    model = TSCNet(2, in_channels=3, use_onehot=False, start_filts=8, depth=5,)

    trainer = Trainer(max_epochs=20, gpus=1, logger=logger,)  # callbacks=[early_stopper]
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

    model.eval()

    # for i, (x, y) in enumerate(train_loader):
    #     output = model(x)
    #     for j, s in enumerate(output):
    #         _, ax = plt.subplots(1, 3)
    #         for a in ax:
    #             a.axis('off')
    #         # ax[0].imshow(hp.inv_normalize(x[j][0, :, :]).permute(1, 2, 0).detach().numpy())
    #         ax[0].imshow(x[j][0].detach().numpy())
    #         ax[1].imshow(output[j].detach().numpy())
    #         ax[2].imshow(y[j][0].detach().numpy())
    #         plt.show()
            # res_path = "/home/joshua/Desktop/Work/SSFCN/experiments/results"
            # z = hp.inv_normalize(x[j][:3, :, :]).permute(1, 2, 0).detach().numpy()
            # z[z>1] = 1
            # plt.imsave(res_path+"/inputs/"+str(2*i+j)+".png", z)
            # plt.imsave(res_path+"/preds/"+str(2*i+j)+".png", output[j].detach().numpy())
            # plt.imsave(res_path+"/targets/"+str(2*i+j)+".png", y[j][0].detach().numpy())

    # train_loader = DataLoader(data, shuffle=True, num_workers=hp.num_workers, batch_size=1)
    #
    # for (x, y) in train_loader:
    #     output = model(x)
    #     _, ax = plt.subplots(1, 3)
    #     for a in ax:
    #         a.axis('off')
    #     ax[0].imshow(hp.inv_normalize(x[0][:3, :, :]).permute(1, 2, 0).detach().numpy())
    #     ax[1].imshow(output[0].detach().numpy())
    #     ax[2].imshow(y[0][0].detach().numpy())
    #     plt.show()


def visualise():
    test_data = DataSet(root_dir="/home/joshua/Desktop/Work/SSFCN/data/4_COCO/validation", transform=hp.train_transform, target_transform=hp.target_transform,
                        concat_coords=True)
    test_data, _ = torch.utils.data.random_split(test_data, [5, len(test_data)-5])
    test_loader = DataLoader(test_data, shuffle=False, num_workers=hp.num_workers, batch_size=1)

    model = TSCNet(4, use_onehot=False, in_channels=5, start_filts=16, depth=8)

    model = model.load_from_checkpoint("/home/joshua/Desktop/Work/SSFCN/experiments/SSFCN-experiments/2re8r63l/checkpoints/epoch=64-step=422564.ckpt")

    model.eval()

    for (x, y) in test_loader:
        output = model(x)

        _, ax = plt.subplots(1, 3)
        for a in ax:
            a.axis('off')
        ax[0].imshow(hp.inv_normalize(x[0][:3, :, :]).permute(1, 2, 0).detach().numpy())
        ax[1].imshow(output[0].detach().numpy())
        ax[2].imshow(y[0][0].detach().numpy())

        # print(torch.nn.CrossEntropyLoss(reduction="mean")(model(x, softmax=False, am=False), y.long()))

        plt.show()

        # print(x.shape)


def display(x):
    _, ax = plt.subplots(1, 2)
    disp = x.permute(0, 2, 3, 1)
    for a in ax:
        a.axis('off')
    ax[0].imshow(disp[0].detach().numpy())
    ax[1].imshow(disp[1].detach().numpy())

    plt.show()


def transpose_test():
    test_data = DataSet(root_dir="/home/joshua/Desktop/Work/SSFCN/data/all_sides_all_perm/train", transform=hp.train_transform,
                        concat_coords=False)
    test_data, _ = torch.utils.data.random_split(test_data, [2, len(test_data)-2])
    test_loader = DataLoader(test_data, shuffle=False, num_workers=hp.num_workers, batch_size=2)

    for (x, y) in test_loader:
        display(x)
        roll_x = x.shape[3]//4
        roll_y = x.shape[2]//4
        for i in range(3):
            rolled_x = torch.roll(x, roll_x+(i*roll_x), 3)
            rolled_y = torch.roll(x, roll_y+(i*roll_y), 2)
            display(rolled_x)


def ds_to_npy():
    idir = "/home/joshua/Desktop/Work/SSFCN/data/7_COCO/validation/inputs/"
    tdir = "/home/joshua/Desktop/Work/SSFCN/data/7_COCO/validation/targets/"

    fdir = "/home/joshua/Desktop/Work/SSFCN/data/4_COCO/validation/"

    curr = 0
    for filename in tqdm(os.listdir("/home/joshua/Desktop/Work/SSFCN/data/7_COCO/validation/inputs")):
        rgb_weights = [0.2989, 0.5870, 0.1140]
        # x = np.array(plt.imread(idir+filename))
        # filename = filename.replace(".jpg", "")
        # y = np.array(plt.imread(tdir+filename+".png"))
        x = np.load(idir+filename)
        y = np.load(tdir+filename)
        # y = y[:, :, :3]
        # gs = np.dot(y[..., :3], rgb_weights)
        gs = y.copy()
        new_mask = gs.copy()
        # unique = np.unique(y.reshape(-1, y.shape[2]), axis=0)
        unique = np.unique(gs)
        num_l = len(unique)

        if num_l <= 4:
            for i, u in enumerate(unique):
                new_mask[gs == u] = i
            new_mask = np.array(new_mask, dtype=np.uint8)
            # _, ax = plt.subplots(1, 3)
            # for a in ax:
            #     a.axis('off')
            # ax[0].imshow(x)
            # ax[1].imshow(new_mask)
            # ax[2].imshow(y)
            # plt.show()
            np.save(fdir+"inputs/"+str(curr)+".npy", x)
            np.save(fdir+"targets/"+str(curr)+".npy", new_mask)
            curr += 1
        else:
            pass
        # if curr == 6:
        #     break


def run_base_dia(dilation, name):
    data = DataSet(root_dir="/home/joshua/Desktop/Work/SSFCN/data/ircad/train", transform=hp.train_transform,
                   target_transform=hp.target_transform, concat_coords=True)
    val_data = DataSet(root_dir="/home/joshua/Desktop/Work/SSFCN/data/ircad/validation", transform=hp.train_transform,
                       target_transform=hp.target_transform, concat_coords=True)
    # test_data = DataSet(root_dir="/home/joshua/Desktop/Work/SSFCN/data/top_square/test", transform=hp.train_transform,
    #                     concat_coords=True)
    # split = int(len(data)*0.99)
    # train_data, val_data = torch.utils.data.random_split(data, [split, len(data)-split])
    # test_data, _ = torch.utils.data.random_split(test_data, [1, len(test_data)-1])
    # data, _ = torch.utils.data.random_split(data, [4, len(data)-4])
    # val_data, _ = torch.utils.data.random_split(val_data, [4, len(val_data)-4])

    train_loader = DataLoader(data, shuffle=True, num_workers=hp.num_workers, batch_size=4)
    val_loader = DataLoader(val_data, shuffle=False, num_workers=hp.num_workers, batch_size=4)
    # test_loader = DataLoader(test_data, shuffle=False, num_workers=hp.num_workers, batch_size=1)

    logger = WandbLogger(name=name,)

    # early_stopper = EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0.00,
    #     patience=5,
    #     verbose=False,
    #     mode='min'
    # )

    model = BNet(2, in_channels=3, use_onehot=False, start_filts=8, depth=5, dilation=dilation)

    trainer = Trainer(max_epochs=20, gpus=1, logger=logger,)  # callbacks=[early_stopper]
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

    # model.eval()
    #
    # for i, (x, y) in enumerate(train_loader):
    #     output = model(x)
    #     for j, s in enumerate(output):
    #         _, ax = plt.subplots(1, 3)
    #         for a in ax:
    #             a.axis('off')
    #         # ax[0].imshow(hp.inv_normalize(x[j][:3, :, :]).permute(1, 2, 0).detach().numpy())
    #         ax[0].imshow(x[j][0].detach().numpy())
    #         ax[1].imshow(output[j].detach().numpy())
    #         ax[2].imshow(y[j][0].detach().numpy())
    #         plt.show()


if __name__ == '__main__':
    # run_base_dia(1, "3-medical-base-20")
    # run_base_dia(2, "3-medical-dia2-20")
    # run_base_dia(3, "3-medical-dia3-20")
    count_labels()
