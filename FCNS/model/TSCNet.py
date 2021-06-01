import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import iou, dice_score
import torch
from torch import nn
from torch.autograd.variable import Variable
import torch.nn.functional as F
from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score as ari
from torch.nn import init
from FCNS.metrics.soft_dice_loss import DiceLoss as DiceLoss
from FCNS.model.utils.class_miou import calculate_class_miou
from FCNS.model.utils.tensorfy import tensorify


def make_one_hot(labels, C=2):
    """
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    """
    if len(labels.shape) == 3:
        labels = labels.unsqueeze(1)
    one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    target = Variable(target)

    return target


def perm_inv(x, y, loss=nn.CrossEntropyLoss()):
    lo1 = loss(x, y)
    inv_y = 1 - y
    lo2 = loss(x, inv_y)
    lo = torch.mean(torch.min(torch.mean(lo1, dim=(1, 2)), torch.mean(lo2, dim=(1, 2))))
    return lo


def g_perm_inv(x, y, C=2, loss=nn.BCELoss(reduction="none")):
    losses = []
    x = x.cpu()
    y = y.cpu()
    for p in permutations(torch.arange(0, C)):
        ny = y.detach().clone()
        for i, c in enumerate(p):
            ny[y == c] = C-(1+i)
        lo1 = loss(x, ny)
        losses.append(lo1)
    losses = torch.stack(losses)
    losses = torch.mean(losses, dim=(2, 3, 4))
    losses = torch.min(losses, dim=0).values
    losses = torch.mean(losses)
    return losses


def eff_perm_inv(x, y, C=2, loss=nn.BCELoss(reduction="none")):
    min_loss = torch.tensor(0)
    # x = x.detach().clone()
    # y = y.detach().clone()
    flag = False
    for p in permutations(torch.arange(0, C)):
        ny = y.detach().clone()
        for i, c in enumerate(p):
            ny[y == c] = C-(1+i)
        # ny = make_one_hot(ny.unsqueeze(1).long(), C=C)
        ny = make_one_hot(ny.long(), C=C)
        # ny = nn.Softmax(dim=1)(ny)
        lo1 = loss(x, ny)
        # losses = torch.stack(losses)
        lo1 = torch.mean(lo1, dim=(1, 2, 3))
        # losses = torch.min(losses, dim=0).values
        # lo1 = torch.mean(lo1, dim=1)
        if flag:
            min_loss = torch.where(min_loss < lo1, min_loss, lo1)
        else:
            min_loss = lo1
            flag = True
    return torch.mean(min_loss)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )

    def forward(self, x):
        out = self.double_conv(x)
        return out


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        self.double_conv = DoubleConv(in_channels, out_channels, dilation)

        self.down_conv = nn.Sequential(
            # nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.PReLU(out_channels)
        )

    def forward(self, x):
        before_down = self.double_conv(x)
        out = self.down_conv(before_down)
        return out, before_down


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(4*in_channels, in_channels, kernel_size=2, stride=2),
            nn.PReLU(in_channels)
        )

        self.double_conv = DoubleConv(in_channels, out_channels, dilation)

    def forward(self, x, xskip, yskip, xyskip):
        if x.shape != xskip.shape:
            x = F.pad(input=x, pad=(0, 1, 0, 1), mode='constant', value=0)

        x = torch.cat((x, xskip, yskip, xyskip), dim=1)
        out = self.up_conv(x)
        out = self.double_conv(out)
        return out


class TSCNet(pl.LightningModule):
    """"

    """
    def __init__(self, num_classes, in_channels=3, depth=5, start_filts=16, dilation=1):
        super(TSCNet, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.depth = depth
        self.start_filts = start_filts
        self.dilation = dilation

        self.down_convs = []
        self.up_convs = []

        self.class_mious = torch.zeros(self.num_classes)
        self.class_samples = torch.zeros(self.num_classes)

        self.final_layer = nn.Conv2d(4*start_filts, num_classes, kernel_size=1)

        self.sm = nn.LogSoftmax(dim=1)

        # Create Encoder
        ins = self.in_channels
        outs = self.start_filts
        for i in range(depth-1):
            down_conv = DownConv(ins, outs, self.dilation)
            self.down_convs.append(down_conv)
            ins = outs
            outs *= 2

        self.middle_conv = DoubleConv(ins, outs, self.dilation)

        # Create Decoder
        ins = outs
        outs //= 2
        for i in range(depth-1):
            up_conv = UpConv(ins, outs, self.dilation)
            self.up_convs.append(up_conv)
            ins = outs
            outs //= 2

        # Initialise Network

        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

        self.save_hyperparameters()

        # self.lo = nn.BCELoss(reduction="none")
        # self.loss = eff_perm_inv
        # self.loss = nn.CrossEntropyLoss(weight=torch.tensor([0.95, 0.05]))
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    @staticmethod
    def weight_init(m) -> None:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, softmax=True, am=True):
        encoder_outs = []

        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        x = self.middle_conv(x)
        encoder_outs.append(x)

        eo_x = []
        eo_y = []
        eo_xy = []

        # This is where the translation for the translated skip connection occurs
        for i, e in enumerate(encoder_outs):
            roll_x = (e.shape[3]//(self.depth+1))*(i+1)
            roll_y = (e.shape[2]//(self.depth+1))*(i+1)

            eo_x.append(torch.roll(e, roll_x, 3))
            eo_y.append(torch.roll(e, roll_y, 2))
            eo_xy.append(torch.roll(torch.roll(e, roll_x, 3), roll_y, 2))

        # This is intended to have normal additive skip and translated skip (F(X)+X, X+, X-, X+-)
        for i, module in enumerate(self.up_convs):
            x = module(x+encoder_outs[-(i+1)], xskip=eo_x[-(i+1)], yskip=eo_y[-(i+1)], xyskip=eo_xy[-(i+1)])

        if x.shape != eo_x[0].shape:
            x = F.pad(input=x, pad=(0, 1, 0, 1), mode='constant', value=0)
        x = self.final_layer(torch.cat((x+encoder_outs[0], eo_x[0], eo_y[0], eo_xy[0]), dim=1))

        if softmax:
            x = self.sm(x)

        if am:
            x = torch.argmax(x, dim=1)

        return x

    def configure_optimizers(self,):
        optimiser = torch.optim.Adadelta(params=self.parameters())
        return optimiser

    def training_step(self, batch, batch_idx):
        x, y = batch

        output = self(x, softmax=False, am=False)
        sm_output = self.sm(output)

        ce = self.ce_loss(output, y.squeeze(1))
        dice = self.dice_loss(sm_output, y)
        total_loss = (ce + dice) / 2

        miou = iou(torch.argmax(sm_output, dim=1), y, self.num_classes)
        dice_s = dice_score(sm_output, y.squeeze(1), bg=True)

        self.log("Train MIoU", miou.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("Train Dice Score", dice_s.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("Train Loss", total_loss.item(), on_step=False, on_epoch=True, prog_bar=False)

        return {'loss': total_loss}

    def validation_step(self, batch, batch_idx) -> object:
        x, y = batch

        output = self(x, softmax=False, am=False)
        sm_output = self.sm(output)

        ce = self.ce_loss(output, y.squeeze(1))
        dice = self.dice_loss(sm_output, y.squeeze(1))
        loss = (ce + dice) / 2

        miou = iou(torch.argmax(sm_output, dim=1), y, self.num_classes)
        dice_s = dice_score(sm_output, y.squeeze(1), bg=True)

        return {'v_loss': loss.item(), 'step_miou': miou.item(), 'v_dice_score': dice_s.item()}

    def validation_epoch_end(self, val_step_outputs) -> None:
        avg_val_loss = torch.tensor([x['v_loss'] for x in val_step_outputs]).mean()
        avg_miou = torch.tensor([x['step_miou'] for x in val_step_outputs]).mean()
        avg_dice = torch.tensor([x['v_dice_score'] for x in val_step_outputs]).mean()

        self.log('Val Loss', avg_val_loss, prog_bar=True)
        self.log('Val MIoU', avg_miou, prog_bar=False)
        self.log('Val Dice Score', avg_dice, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch

        output = self(x, softmax=False, am=False)
        sm_output = self.sm(output)

        cms, css = calculate_class_miou(sm_output, y, self.num_classes)

        self.class_mious += cms
        self.class_samples += css

    def test_epoch_end(self, outputs):
        self.class_mious = torch.where(self.class_samples == torch.scalar_tensor(0.0), torch.scalar_tensor(0.0), self.class_mious / self.class_samples)

        # self.class_mious /= torch.scalar_tensor(self.class_samples)

        for i, cm in enumerate(self.class_mious):
            self.log("Class " + str(i), self.class_mious[i])
