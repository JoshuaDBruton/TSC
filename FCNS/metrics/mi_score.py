import torch
from torch.autograd import Variable


def coo(X, Y):
    """
    Author: Joshua Bruton
    Finds the contingency matrix given two discrete distributions
    :param X: DoubleTensor of shape (H, W) or (B, ..., H, W) where ... represents any number of dimensions
    :param Y: DoubleTensor of shape (H, W) or (B, ..., H, W) where ... represents any number of dimensions
    :returns: 2D DoubleTensor of shape (XC, YC) where XC and YC are the number of classes in X and Y, respectively
    """
    torch.set_default_dtype(torch.float32)

    # If batches are used, the entire batch is averages into one
    if len(X.shape) > 2:
        b = X.shape[0]
        X = torch.sum(X, dim=0)/b
        Y = torch.sum(Y, dim=0)/b
    X = torch.flatten(X)
    Y = torch.flatten(Y)
    labels_X = torch.unique(X)
    labels_Y = torch.unique(Y)
    cx = torch.tensor(len(labels_X))
    cy = torch.tensor(len(labels_Y))
    n = torch.tensor(len(X))

    matrix = torch.zeros((cx, cy))

    for i in range(cx):
        for j in range(cy):
            matrix[i][j] = torch.count_nonzero(torch.logical_and(X == labels_X[i], Y == labels_Y[j]))

    return matrix/n


def mi_score(X, Y):
    """
    Author: Joshua Bruton
    Warning: This function is not differentiable and so cannot be used as a loss function
    :param X: DoubleTensor of shape (H, W) or (B, ..., H, W) where ... represents any number of dimensions
    :param Y: DoubleTensor of shape (H, W) or (B, ..., H, W) where ... represents any number of dimensions
    :return: The mutual information score of the distributions X and Y (average/sum of all batches)
    """
    X = X.float()
    Y = Y.float()
    cm = coo(X, Y)

    pxs = torch.sum(cm, dim=1)
    pys = torch.sum(cm, dim=0)

    pmarg = torch.flatten(torch.matmul(pxs.reshape(-1, 1), pys.reshape(1, -1)))

    fcm = torch.flatten(cm)+torch.tensor(1e-14)

    lmarg = torch.log(fcm/pmarg)

    return torch.sum(torch.dot(fcm, lmarg))
