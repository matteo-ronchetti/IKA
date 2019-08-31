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


def spherical_grid(d, n_angles, n_radii):
    radii = [x / (n_radii - 1) for x in range(n_radii)]
    if d == 1:
        grid = np.concatenate([np.asarray(radii), -np.asarray(radii)]).reshape(-1, 1)
    else:

        angles = np.arange(n_angles) * 2 * np.pi / n_angles
        angles = np.hstack([c.reshape(-1, 1) for c in np.meshgrid(*[angles] * (d - 1), sparse=False)])
        # print(angles / np.pi)

        a = np.hstack((np.array([2 * np.pi] * angles.shape[0]).reshape(-1, 1), angles))
        si = np.sin(a)
        si[:, 0] = 1
        si = np.cumprod(si, axis=1)
        co = np.cos(a)
        co = np.roll(co, -1, axis=1)
        points = si * co

        grid = [np.zeros(d).reshape(1, -1)]
        for r in radii:
            grid.append(r * points)

        grid = np.vstack(grid)

    # remove duplicate points
    o = np.ones((grid.shape[0], grid.shape[0]))
    d = distance_matrix(grid, grid) + np.triu(o)
    good = np.min(d, axis=1) > 1e-6
    return grid[good]


class GaussianExpectedValueEstimator:
    def __init__(self, kernel_sigma, sigma):
        self.kernel_sigma = kernel_sigma
        self.sigma = sigma

    def compute_grid(self, dim, angles, radii, optimize_iter=3000):
        grid = spherical_grid(dim, angles, radii)
        x = Variable(torch.FloatTensor(grid), requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=3e-3)

        with torch.no_grad():
            D = torch_distance_matrix(x, x, squared=True)
            S = torch.sum(x ** 2, dim=1)
            print(f"Distance range {(torch.min(D + torch.eye(x.size(0)) * 1e30).item(), torch.max(D).item())}")

        def f(k):
            with torch.no_grad():
                G = torch.exp(-0.5 * k ** 2 * D / self.kernel_sigma)
                c = torch.exp(-(0.5 * k ** 2 * S) / (self.sigma + self.kernel_sigma)).view(-1, 1)

                y, _ = torch.solve(c, G)
                return (-c.t() @ y).item()

        k, v = dlib.find_min_global(f, [0.1], [10.0], 50)
        k = k[0]
        print("Grid scale:", k)

        x.data *= k

        if optimize_iter > 0:
            optimizer.zero_grad()
            for _ in tqdm(range(optimize_iter)):
                G = torch.exp(-0.5 * torch_distance_matrix(x, x, squared=True) / self.kernel_sigma)
                c = torch.exp(-(0.5 * torch.sum(x ** 2, dim=1)) / (self.sigma + self.kernel_sigma)).view(-1, 1)

                y, _ = torch.solve(c, G)
                loss = - c.t() @ y

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return x.data.numpy()

    def quadrature_weights(self, grid):
        G = np.exp(-0.5 * distance_matrix(grid, grid) ** 2 / self.kernel_sigma)
        c = np.exp(-(0.5 * np.linalg.norm(grid, axis=1) ** 2) / (self.kernel_sigma + self.sigma))

        c *= (self.kernel_sigma / (self.kernel_sigma + self.sigma)) ** (grid.shape[1] / 2)

        w, _ = cg(G, c)

        return w

    def estimate(self, grid, f):
        G = np.exp(-0.5 * distance_matrix(grid, grid) ** 2 / self.kernel_sigma)
        c = np.exp(-(0.5 * np.linalg.norm(grid, axis=1) ** 2) / (self.kernel_sigma + self.sigma))

        c *= (self.kernel_sigma / (self.kernel_sigma + self.sigma)) ** (grid.shape[1] / 2)

        w, _ = cg(G, c)

        return float(np.inner(w, f(grid)))


def load_pretrained(path, device):
    ika_features = nn.Sequential(
        HardNet(),
        nn.Linear(128, 128),
        nn.Linear(128, 1024 * 4),
        Exp()
    )

    data = torch.load(path, map_location=device)
    ika_features.load_state_dict(data["features"])
    ika_features.eval()
    model = IKA(ika_features)
    model.linear = data["ika"]
    model.eval()
    model = model.to(device)
    return model


def points_to_transformation(x):
    # 1 = 10°
    return torch.cat([SpatialTransformation.compute_grid(32, 32, translation=(0, 0, 0), rot=(p[0] * 0.174533, 0, 0),
                                                         scale=1).unsqueeze(0) for p in x])


def kernel_interpolation_error(model, train_points, test_points, x, device):
    train_grid = points_to_transformation(train_points).to(device)
    test_grid = points_to_transformation(test_points).to(device)

    D_train = torch.FloatTensor(distance_matrix(train_points, train_points) ** 2).to(device)
    D_test = torch.FloatTensor(distance_matrix(test_points, train_points) ** 2).to(device)

    def f(kernel_sigma):
        with torch.no_grad():
            y_train = model(F.grid_sample(x.repeat(train_grid.size(0), 1, 1, 1), train_grid, padding_mode="border"))
            y_test = model(F.grid_sample(x.repeat(test_grid.size(0), 1, 1, 1), test_grid, padding_mode="border"))

            G = torch.exp(-0.5 * D_train / kernel_sigma)
            G2 = torch.exp(-0.5 * D_test / kernel_sigma)

            w, _ = torch.solve(y_train, G)
            y_ = (G2 @ w)

            loss = torch.mean((y_test - y_) ** 2).item()
            return loss

    return f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="liberty")
    parser.add_argument("--sampling-points", default=25, type=int)
    parser.add_argument("--gram-size", default=60000, type=int)
    parser.add_argument("--sigma", default=1.0, type=float)
    parser.add_argument("--model", default="models/model_4096.pth")
    parser.add_argument("--output", default="integrated.npz")
    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f"Found GPU {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    else:
        device = torch.device("cpu")

    model = load_pretrained(args.model, device)

    # load and shuffle dataset
    dataset = PhotoTour("data/phototour", "liberty", download=True, train=True)
    X = dataset.data.unsqueeze(1)
    X = X[torch.randperm(X.size(0))]

    # compute the first grid used to choose kernel sigma
    estimator = GaussianExpectedValueEstimator(0.25, args.sigma)
    train_points = estimator.compute_grid(1, 0, args.sampling_points, optimize_iter=0)
    print("Grid has", train_points.shape[0], "points")
    test_points = np.random.randn(200, 1)

    # find a good kernel sigma
    x = X[0].to(device).float() / 255
    interpolation_error_f = kernel_interpolation_error(model, train_points, test_points, x, device)
    kernel_sigma = dlib.find_min_global(interpolation_error_f, [0.01], [4.0], 50)[0][0]
    print("Kernel Sigma:", kernel_sigma)
    estimator.kernel_sigma = kernel_sigma

    # compute and optimize the grid of points
    grid = estimator.compute_grid(1, 0, args.sampling_points, optimize_iter=3000)
    transformations = points_to_transformation(grid).to(device)

    # compute quadrature weights
    w = torch.FloatTensor(estimator.quadrature_weights(grid)).to(device)
    print(grid)
    print(w)

    ts = transformations.size(0)
    bs = 1 + 500 // transformations.size(0)
    size = (args.gram_size // bs) * bs

    print(f"Size: {size}, Batch size: {bs}")
    dataloader = DataLoader(TensorDataset(X[:size]), bs, shuffle=False, pin_memory=True, drop_last=False)
    transformations = transformations.repeat(bs, 1, 1, 1).contiguous()

    Y = torch.empty((size, 4096)).to(device)
    w = w.unsqueeze(0).repeat(bs, 1).view(bs, 1, -1)
    print(w)
    i = 0
    for x, in tqdm(dataloader):
        x = x.float().to(device) / 255
        with torch.no_grad():
            tmp = F.grid_sample(x.repeat_interleave(ts, dim=0), transformations, padding_mode="border")
            # print(tmp.size())
            y = model(tmp).view(bs, -1, 4096)
            # print(y.size(), w.size())
            Y[i:i + bs] = torch.bmm(w, y).squeeze(1)
            i += bs
            if i >= size:
                break
    print(i, "==", size)
    np.savez(args.output, phi=Y.cpu().data.numpy(), X=X.cpu().data.numpy())


if __name__ == "__main__":
    main()
