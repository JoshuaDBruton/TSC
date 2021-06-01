import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import iou, dice_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from FCNS.metrics.soft_dice_loss import DiceLoss as DiceLoss
from FCNS.model.utils.class_miou import calculate_class_miou
from FCNS.model.utils.tensorfy import tensorify


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, bias=bias, groups=groups
                     , dilation=dilation)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    else:
        return nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2), conv1x1(in_channels, out_channels))


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, dilation, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels, dilation=dilation)
        self.conv2 = conv3x3(self.out_channels, self.out_channels, dilation=dilation)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation,
                 merge_mode='concat',
                 up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels,
                                self.out_channels,
                                mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(2*self.out_channels, self.out_channels, dilation=dilation)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels, dilation=dilation)

        self.conv2 = conv3x3(self.out_channels, self.out_channels, dilation=dilation)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet(pl.LightningModule):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, num_classes, dilation, in_channels=3, depth=5,
                 start_filts=64, up_mode='transpose',
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.dilation = dilation

        self.down_convs = []
        self.up_convs = []

        self.class_mious = torch.zeros(self.num_classes)
        self.class_samples = 0

        self.sm = nn.LogSoftmax(dim=1)
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False

            down_conv = DownConv(ins, outs, pooling=pooling, dilation=self.dilation)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode, merge_mode=merge_mode, dilation=self.dilation)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

        self.save_hyperparameters()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, softmax=True, am=True):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        x = self.conv_final(x)

        if softmax:
            x = self.sm(x)

        if am:
            x = torch.argmax(x, dim=1)

        return x

    def configure_optimizers(self):
        optimiser = torch.optim.Adadelta(params=self.parameters())
        return optimiser

    def training_step(self, batch, batch_idx):
        x, y = batch

        output = self(x, softmax=False, am=False)
        sm_output = self.sm(output)

        ce = self.ce_loss(output, y.squeeze(1))
        dice = self.dice_loss(sm_output, y)
        total_loss = (ce+dice)/2

        miou = iou(torch.argmax(sm_output, dim=1), y, self.num_classes)
        dice_s = dice_score(sm_output, y.squeeze(1), bg=True)

        self.log("Train MIoU", miou.item(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("Train Dice Score", dice_s.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("Train Loss", total_loss.item(), on_step=False, on_epoch=True, prog_bar=False)

        return {'loss': total_loss}

    def validation_step(self, batch, batch_idx) -> object:
        x, y = batch

        output = self(x, softmax=False, am=False)
        sm_output = self.sm(output)

        ce = self.ce_loss(output, y.squeeze(1))
        dice = self.dice_loss(sm_output, y.squeeze(1))
        loss = (ce+dice)/2

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
