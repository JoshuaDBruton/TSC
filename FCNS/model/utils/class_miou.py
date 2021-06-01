import torch
from pytorch_lightning.metrics.functional import iou, dice_score


def calculate_class_miou(output, y, num_classes):
    class_mious = torch.zeros(num_classes)
    class_samples = torch.zeros(num_classes)

    output = output.detach().clone()
    output = torch.argmax(output, dim=1).unsqueeze(1)
    for i in range(num_classes):
        a = output.detach().clone()
        b = y.detach().clone()
        a[output != i] = 0
        a[output == i] = 1
        b[y != i] = 0
        b[y == i] = 1

        class_mious[i] = iou(a[:, 0, :, :], b, 2).item()
        if i in torch.unique(y):
            class_samples[i] += 1
        else:
            class_mious[i] = 0.0

    return class_mious, class_samples
