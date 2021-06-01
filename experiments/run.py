from experiments.dataset.image_pair import ImageSegmentation as DataSet
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from FCNS.model.BNet import BNet
from FCNS.model.TSCNet import TSCNet
from FCNS.model.fcn import UNet
from pytorch_lightning import Trainer
import os
from torchvision import transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt
from experiments.project_name import ProjectName as conf

ROOT_DATA_PATH = "/home/joshua/Desktop/data"

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

INV_NORMALISE = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
)


def run(network, dataset_name, experiment_name, epochs, num_classes, depth, start_filts, dilation):
    data_path = os.path.join(ROOT_DATA_PATH, dataset_name)
    data = DataSet(root_dir=os.path.join(data_path, "train"), transform=TRAIN_TRANSFORM,
                   target_transform=TARGET_TRANSFORM, concat_coords=conf.concat_coords,)
    val_data = DataSet(root_dir=os.path.join(data_path, "validation"), transform=TRAIN_TRANSFORM,
                       target_transform=TARGET_TRANSFORM, concat_coords=conf.concat_coords,)
    # data, _ = torch.utils.data.random_split(data, [4, len(data) - 4])
    # val_data, _ = torch.utils.data.random_split(val_data, [4, len(val_data) - 4])

    train_loader = DataLoader(data, shuffle=True, num_workers=4, batch_size=conf.batch_size)
    val_loader = DataLoader(val_data, shuffle=False, num_workers=4, batch_size=conf.batch_size)
    logger = WandbLogger(project=conf.project_name, name=experiment_name)

    in_channels = 5 if conf.concat_coords else 3

    model = None
    if network == "bnet":
        model = BNet(num_classes, in_channels=in_channels, start_filts=start_filts, depth=depth, dilation=dilation)
    elif network == "tsc":
        model = TSCNet(num_classes, in_channels=in_channels, start_filts=start_filts, depth=depth, dilation=dilation)
    elif network == "unet":
        model = UNet(num_classes, in_channels=in_channels, start_filts=start_filts, depth=depth, dilation=dilation)

    trainer = Trainer(max_epochs=epochs, gpus=1, logger=logger,)  # callbacks=[early_stopper]
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

    if conf.run_test_loop:
        test_loader = DataLoader(val_data, shuffle=False, num_workers=4, batch_size=1)
        trainer.test(test_dataloaders=test_loader)

    # model.eval()
    #
    # for i, (x, y) in enumerate(train_loader):
    #     output = model(x)
    #     for j, s in enumerate(x):
    #         _, ax = plt.subplots(1, 3)
    #         for a in ax:
    #             a.axis('off')
    #         print(torch.unique(output[j]))
    #         print(torch.unique(y[j][0]))
    #         print()
    #         ax[0].imshow(INV_NORMALISE(x[j][:3, :, :]).permute(1, 2, 0).detach().numpy())
    #         # ax[0].imshow(x[j][0].detach().numpy())
    #         ax[1].imshow(output[j].detach().numpy())
    #         ax[2].imshow(y[j][0].detach().numpy())
    #         plt.show()


if __name__ == "__main__":
    run(network="unet", dataset_name=conf.dataset_name, experiment_name=conf.experiment_prefix+"-unet", epochs=conf.epochs, num_classes=conf.num_classes, depth=conf.depth, start_filts=conf.start_filts, dilation=1)
    # run(network="unet", dataset_name=conf.dataset_name, experiment_name=conf.experiment_prefix + "-unet-di2", epochs=conf.epochs, num_classes=conf.num_classes, depth=conf.depth, start_filts=conf.start_filts, dilation=2)
    # run(network="unet", dataset_name=conf.dataset_name, experiment_name=conf.experiment_prefix + "-unet-di3", epochs=conf.epochs, num_classes=conf.num_classes, depth=conf.depth, start_filts=conf.start_filts, dilation=3)
    # run(network="bnet", dataset_name=conf.dataset_name, experiment_name=conf.experiment_prefix+"-bnet", epochs=conf.epochs, num_classes=conf.num_classes, depth=conf.depth, start_filts=conf.start_filts, dilation=1)
    # run(network="bnet", dataset_name=conf.dataset_name, experiment_name=conf.experiment_prefix+"-bnet-di2", epochs=conf.epochs, num_classes=conf.num_classes, depth=conf.depth, start_filts=conf.start_filts, dilation=2)
    # run(network="bnet", dataset_name=conf.dataset_name, experiment_name=conf.experiment_prefix+"-bnet-di3", epochs=conf.epochs, num_classes=conf.num_classes, depth=conf.depth, start_filts=conf.start_filts, dilation=3)
    # run(network="tsc", dataset_name=conf.dataset_name, experiment_name=conf.experiment_prefix+"-tsc", epochs=conf.epochs, num_classes=conf.num_classes, depth=conf.depth, start_filts=conf.start_filts, dilation=1)
