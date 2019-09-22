import argparse

import dlib

import numpy as np
import scipy.optimize
from scipy.spatial import distance_matrix
from ika import distance_matrix as torch_distance_matrix
from ika import IKA
from scipy.sparse.linalg import cg
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from architectures.layers import Exp
from architectures.hardnet import HardNet
from torchvision.datasets import PhotoTour
from torch.utils.data import DataLoader, TensorDataset
from transformation import TransformPipeline, SpatialTransformation

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


class Integrator:
    def __init__(self, sample_function, sample_points, n_points, dist_sigma, batch_size=128, n_functions=16,
                 n_iter=1000):
        S = torch.linspace(-1, 1, n_points).view(-1, 1)

        x = sample_points(batch_size)
        fs = [sample_function() for _ in range(n_functions)]

        with torch.no_grad():
            D = torch_distance_matrix(S, S, squared=True)

        def ff(s, ss):
            loss = 0
            with torch.no_grad():
                for f in fs:
                    y = f(x)
                    ys = f(s * S)

                    G = torch.exp(-0.5 * D * s ** 2 / ss)
                    Gv = torch.exp(-0.5 * torch_distance_matrix(x, s * S, squared=True) / ss)
                    try:
                        w, _ = torch.solve(ys, G)
                    except Exception as e:
                        print(e)
                        print(s, ss)
                    y_ = Gv @ w
                    loss += torch.mean((y - y_)).item() ** 2
            return loss

        (s, sigma), _ = dlib.find_min_global(ff, [0.01, 0.01], [10.0, 10.0], 500)
        print("Scale:", s, " Sigma:", sigma)

        sigma = Variable((torch.ones(n_points) * sigma), requires_grad=True)
        S *= s
        S = Variable(S, requires_grad=True)
        optimizer = torch.optim.Adam([S, sigma])

        tot_loss = 0
        for i in range(n_iter):
            with torch.no_grad():
                x = sample_points(batch_size)
                f = sample_function()
                y = f(x)
                ys = f(S)

            G = torch.exp(-0.5 * torch_distance_matrix(S, S, squared=True) / sigma[None, :])
            Gv = torch.exp(-0.5 * torch_distance_matrix(x, S, squared=True) / sigma[None, :])
            w, _ = torch.solve(ys, G)
            y_ = Gv @ w
            loss = torch.mean((y - y_) ** 2)

            tot_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(tot_loss)
                tot_loss = 0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.S = S.data
        self.sigma = sigma.data
        with torch.no_grad():
            # w = np.sqrt(2 * 3.14159) * self.sigma
            print("sigma:", self.sigma, " S:", self.S)
            q = torch.exp(-(0.5 * torch.sum(self.S ** 2, dim=1)) / (self.sigma + dist_sigma))
            q *= (dist_sigma / (dist_sigma + self.sigma)) ** (1.0 / 2)
            G = torch.exp(-0.5 * torch_distance_matrix(S, S, squared=True) / sigma[None, :]).t()
            self.w, _ = torch.solve(q.view(-1, 1), G)
            print(self.w)

    def predict(self, f, x):
        with torch.no_grad():
            y = f(self.S)
            G = torch.exp(-torch_distance_matrix(self.S, self.S, squared=True) / self.sigma.clamp(1e-4))
            Gv = torch.exp(-torch_distance_matrix(x, self.S, squared=True) / self.sigma.clamp(1e-4))
            w, _ = torch.solve(y, G)
            return Gv @ w

    def integrate(self, f):
        with torch.no_grad():
            y = f(self.S)
            return y.t() @ self.w


def main():
    def sample_function(s=None):
        if s is None:
            s = np.random.uniform(-1, 1)

        def f(x):
            # return torch.sign(x)*0.2
            return np.exp(-np.sum((x + s) ** 2, axis=1) / 2)
            # return torch.exp(-torch.sum((x + s) ** 2, dim=1) / 2).view(-1, 1)  # + torch.sign(x)

        return f

    def sample_points(n):
        return np.random.randn(n, 1)

    S = np.linspace(-1, 1, 7).reshape(-1, 1)
    print(S)
    y = []
    F = []

    for i in range(10000):
        f = sample_function()
        F.append(f(S))
        y.append(np.mean(f(sample_points(128))))

    F = np.vstack(F)
    y = np.asarray(y)

    w, _, _, _ = np.linalg.lstsq(F, y, rcond=None)
    print(w)

    err = 0
    for i in range(100):
        s = np.random.uniform(-1, 1)
        f = sample_function(s)

        err += np.abs(np.dot(w, f(S).reshape(-1)) - np.exp(-s ** 2 / 4) / np.sqrt(2))

    print(err)

    # integrator = Integrator(sample_function, sample_points, 7, 1.0, n_iter=10000)
    #
    # x = sample_points(10000)
    #
    # # x = torch.linspace(-4, 4, 400).view(-1, 1)
    # # y = f(x)
    # # y_ = integrator.predict(f, x)
    #
    # for s in [0, 0.5, 1]:
    #     f = sample_function(s)
    #     print(s, integrator.integrate(f).item(), np.exp(-s ** 2 / 4) / np.sqrt(2), torch.mean(f(x)))
    #     print()

    # plt.plot(x.numpy(), y.numpy())
    # plt.plot(x.numpy(), y_.numpy())
    # plt.show()


main()
