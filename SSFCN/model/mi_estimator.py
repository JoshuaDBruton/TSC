import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math

EPS = 1e-6


class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ins, running_ema):
        ctx.save_for_backward(ins, running_ema)
        input_log_sum_exp = ins.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        ins, running_mean = ctx.saved_tensors
        grad = grad_output * ins.exp().detach() / (running_mean + EPS) / ins.shape[0]
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    return t_log, running_mean


class ConcatLayer(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x, y):
        return torch.cat((x, y), self.dim)


class CustomSequential(nn.Sequential):
    def forward(self, *ins):
        for module in self._modules.values():
            if isinstance(ins, tuple):
                ins = module(*ins)
            else:
                ins = module(ins)
        return ins


class T(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim=400):
        super().__init__()
        self.layers = CustomSequential(ConcatLayer(), nn.Linear(x_dim + z_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, 1))

    def forward(self, x, z):
        x = torch.flatten(x, start_dim=1)
        z = torch.flatten(z, start_dim=1)
        return self.layers(x.float(), z.float())


class Mine(nn.Module):
    def __init__(self, mi_model, loss='mine', alpha=0.01):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha

        self.T = mi_model
        self.sm = nn.Sigmoid()

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        t = self.T(x, z).mean()
        t_marg = self.T(x, z_marg)

        second_term, self.running_mean = ema_loss(t_marg, self.running_mean, self.alpha)

        return -t + second_term

    def mi(self, x, z, z_marg=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()

        with torch.no_grad():
            mi = -self.forward(x, z, z_marg)
        return mi
