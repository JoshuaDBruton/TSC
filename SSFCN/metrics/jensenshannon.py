# from torch.linalg import norm
# import torch.nn.functional as F
#
#
# def jensen_shannon(x, z, reduction="batchmean"):
#     """
#     Author: Joshua Bruton
#     :param x: FloatTensor of shape (B, N) where B denotes the batch size
#     :param z: FloatTensor of shape (B, N) where B denotes the batch size
#     :param reduction: The type of reduction to use, batchmean by default
#     :return: The jensen-shannon distance between x and y, reduced from kl_div using batchmean by default
#     """
#     # p = x / norm(x, ord=1, keepdim=True, dim=1)
#     # q = z / norm(z, ord=1, keepdim=True, dim=1)
#
#     m = 0.5 * (x + z)
#
#     return 0.5 * (F.kl_div(m.log(), x, reduction=reduction) + F.kl_div(m.log(), z, reduction=reduction))
