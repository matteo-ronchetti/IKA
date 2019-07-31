import torch
import torch.nn as nn
import scipy.linalg
import numpy as np


def distance_matrix(x, y, squared=False):
    x_flat = x.view(-1, x.size(-1))

    x_norm = (x_flat ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x_flat, y.t())
    dist = dist.clamp(0)

    if not squared:
        dist = torch.sqrt(dist)

    return dist.view(*x.size()[:-1], -1)


class IKA(nn.Module):
    def __init__(self, b):
        super().__init__()
        self.b = b
        self.linear = None

    def forward(self, x):
        B = self.b(x)
        return B @ self.linear

    def compute_linear_layer(self, X, G, eps=0, rank=-1, min_lambda=0):
        with torch.no_grad():
            B = self.b(X)  # .cpu().data.numpy()

            if eps == 0:
                Q, R = torch.qr(B)
                r = B.size(1)
                p = None
            else:
                B_norms = np.linalg.norm(B, axis=0)
                Q, R, p = scipy.linalg.qr(B, mode="economic", pivoting=True)

                residual = 1 - np.abs(np.diag(R)) / B_norms[p]
                r = np.searchsorted(residual, 1 - eps)

                Q = torch.FloatTensor(Q[:, :r]).to(G.device)
                R = R[:r, :r]

            M = (Q.t() @ (G @ Q)).cpu().data.numpy()

            # computes d and V
            d, V = scipy.linalg.eigh(M)
            d = np.sqrt(np.maximum(d, 0))
            V = scipy.linalg.solve_triangular(R, V)

            # choose approximation rank
            if rank == -1:
                rank = Q.shape[1]
            rank = min(rank, Q.shape[1] - np.searchsorted(d, min_lambda))

            psi = V[:, -rank:] @ np.diag(d[-rank:])

            # put zeros corresponding to the dropped functions
            if r != B.size(1):
                psi_ = np.zeros((B.size(1), rank), dtype=np.float32)
                psi_[p] = psi
                psi = psi_

            self.linear = nn.Parameter(torch.FloatTensor(psi))
