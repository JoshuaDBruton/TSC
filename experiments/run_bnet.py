import torch
from torch.utils.data import DataLoader
from FCNS.model.BNet import BNet
from pytorch_lightning import Trainer
from experiments.conf import HyperParameters as hp
from experiments.dataset.pair import ImageSegmentation as DataSet
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt


def run():
    data = DataSet(root_dir="/home/joshua/Desktop/Work/SSFCN/data/4_COCO/train", transform=hp.train_transform,
                   target_transform=hp.target_transform, concat_coords=True)
    val_data = DataSet(root_dir="/home/joshua/Desktop/Work/SSFCN/data/4_COCO/validation", transform=hp.train_transform,
                       target_transform=hp.target_transform, concat_coords=True)
    # test_data = DataSet(root_dir="/home/joshua/Desktop/Work/FCNS/data/top_square/test", transform=hp.train_transform,
    #                     concat_coords=True)
    # split = int(len(data)*0.99)
    # train_data, val_data = torch.utils.data.random_split(data, [split, len(data)-split])
    # test_data, _ = torch.utils.data.random_split(test_data, [1, len(test_data)-1])
    data, _ = torch.utils.data.random_split(data, [4, len(data)-4])
    val_data, _ = torch.utils.data.random_split(val_data, [4, len(val_data)-4])

    train_loader = DataLoader(data, shuffle=True, num_workers=hp.num_workers, batch_size=4)
    val_loader = DataLoader(val_data, shuffle=False, num_workers=hp.num_workers, batch_size=4)
    # test_loader = DataLoader(test_data, shuffle=False, num_workers=hp.num_workers, batch_size=1)

    # logger = WandbLogger(name="coco-base-20",)

    # early_stopper = EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0.00,
    #     patience=5,
    #     verbose=False,
    #     mode='min'
    # )

    model = BNet(4, in_channels=5, use_onehot=False, start_filts=8, depth=5, dilated=False)

    trainer = Trainer(max_epochs=50, gpus=1,)  # logger=logger, callbacks=[early_stopper]
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

    model.eval()

    for i, (x, y) in enumerate(train_loader):
        output = model(x)
        for j, s in enumerate(output):
            _, ax = plt.subplots(1, 3)
            for a in ax:
                a.axis('off')
            ax[0].imshow(hp.inv_normalize(x[j][:3, :, :]).permute(1, 2, 0).detach().numpy())
            ax[1].imshow(output[j].detach().numpy())
            ax[2].imshow(y[j][0].detach().numpy())
            plt.show()


if __name__ == '__main__':
    run()
