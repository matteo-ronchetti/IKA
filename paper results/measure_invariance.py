import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from architectures.layers import Exp
from architectures.hardnet import HardNet
from ika import IKA
from ika import distance_matrix as torch_distance_matrix
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt
import dlib
from torchvision.datasets import PhotoTour
from torch.utils.data import DataLoader
from transformation import TransformPipeline, SpatialTransformation

from integration import GaussianExpectedValueEstimator, points_to_transformation, load_pretrained


def load_model(path, device):
    ika_features = nn.Sequential(
        HardNet(),
        nn.Linear(128, 128),
        nn.Linear(128, 4096),
        Exp()
    )

    data = torch.load(path, map_location=device)
    ika_features.load_state_dict(data["features"])
    ika_features.eval()

    model = IKA(ika_features)
    model.linear = data["ika"]
    return model


def k(model, x, y):
    return torch.exp(-torch_distance_matrix(model(x), model(y), squared=True) / 2)


def kernel_interpolation_error(model, train_points, test_points, x, device):
    train_grid = points_to_transformation(train_points).to(device)
    test_grid = points_to_transformation(test_points).to(device)

    D_train = torch.FloatTensor(distance_matrix(train_points, train_points) ** 2).to(device)
    D_test = torch.FloatTensor(distance_matrix(test_points, train_points) ** 2).to(device)

    T = TransformPipeline(SpatialTransformation(dst_size=(32, 32)))

    def f(kernel_sigma):
        with torch.no_grad():
            y_train = k(model,
                        F.grid_sample(x.repeat(train_grid.size(0), 1, 1, 1), train_grid, padding_mode="border"), T(x))
            y_test = k(model,
                       F.grid_sample(x.repeat(test_grid.size(0), 1, 1, 1), test_grid, padding_mode="border"), T(x))

            G = torch.exp(-0.5 * D_train / kernel_sigma)
            G2 = torch.exp(-0.5 * D_test / kernel_sigma)

            w, _ = torch.solve(y_train, G)
            y_ = (G2 @ w)

            loss = torch.mean((y_test - y_) ** 2).item()
            return loss

    return f


def main():
    ys = np.load("tmp.npy")
    sigma = 2.0 * 0.174533 * 30

    radius = 1
    for i in range(2, 50):
        y = math.exp(-(i ** 2) / (4 * sigma ** 2))
        radius = i - 1
        if y <= 0.01:
            break

    print(radius)
    filt = torch.arange(-radius, radius + 1, dtype=torch.float32)
    filt = torch.exp(-filt.pow(2) / (4. * sigma * sigma))
    filt /= torch.sum(filt)
    filt = filt.numpy()

    plt.plot(ys)
    zs = np.convolve(np.pad(ys, (radius, radius), mode="edge"), filt, mode="valid")
    zs /= np.max(zs)
    plt.plot(zs)
    plt.show()

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", default="liberty")
    # parser.add_argument("--model", nargs="+", default="models/checkpoint_liberty_no_aug.pth")
    # parser.add_argument("--sampling-points", default=5, type=int)
    #
    # args = parser.parse_args()
    #
    # if torch.cuda.is_available():
    #     print(f"Found GPU {torch.cuda.get_device_name(0)}")
    #     device = torch.device('cuda')
    # else:
    #     device = torch.device("cpu")
    #
    # dataset = PhotoTour("data/phototour", args.dataset, download=True, train=False)
    # X = dataset.data[:100].unsqueeze(1).float().to(device) / 255
    #
    # for p in args.model:
    #     with torch.no_grad():
    #         m = load_model(p, device)
    #         center = F.normalize(m(TransformPipeline(SpatialTransformation(dst_size=(32, 32)))(X)))
    #
    #         ys = []
    #
    #         for alpha in range(-60, 61):
    #             T = TransformPipeline(SpatialTransformation(rot=(alpha / 30, 0, 0), dst_size=(32, 32)))
    #             t = F.normalize(m(T(X)))
    #             ys.append(torch.mean(torch.sum(center * t, dim=1)).item())
    #
    #     ys = np.asarray(ys)
    #     np.save("tmp.npy", ys)
    # ws = 51
    # plt.plot(ys, label=p)
    # plt.plot(np.convolve(np.pad(ys, (ws//2, ws//2), mode="edge"), np.ones((ws,))/ws, mode="valid"))

    # model = HardNet.from_file(args.model, device)
    # model.eval()
    #
    # dataset = PhotoTour("data/phototour", args.dataset, download=True, train=False)
    # X = dataset.data[:100].unsqueeze(1).float().to(device) / 255
    # T = TransformPipeline(SpatialTransformation(dst_size=(32, 32)))
    #
    # sigma = 1**2
    # estimator = GaussianExpectedValueEstimator(0.25, sigma)
    # train_points = estimator.compute_grid(1, 0, args.sampling_points, optimize_iter=0)
    # print("Grid has", train_points.shape[0], "points")
    # test_points = np.random.randn(50, 1)*sigma
    #
    # # find a good kernel sigma
    # x = X[0].unsqueeze(0)
    # interpolation_error_f = kernel_interpolation_error(model, train_points, test_points, x, device)
    # kernel_sigma = dlib.find_min_global(interpolation_error_f, [0.01], [4.0], 50)[0][0]
    # print("Kernel Sigma:", kernel_sigma)
    # estimator.kernel_sigma = kernel_sigma
    #
    # # compute and optimize the grid of points
    # grid = estimator.compute_grid(1, 0, args.sampling_points, optimize_iter=3000)
    # transformations = points_to_transformation(grid).to(device)
    #
    # grid = np.hstack([c.reshape(-1, 1) for c in np.meshgrid(grid, grid, sparse=False)])
    # w = estimator.quadrature_weights(grid).reshape(transformations.size(0), transformations.size(0))
    # print(w.shape)
    # w = torch.FloatTensor(w)
    #
    # def K(x, y):
    #     tx = F.grid_sample(x.repeat(transformations.size(0), 1, 1, 1), transformations, padding_mode="border")
    #     ty = F.grid_sample(y.repeat(transformations.size(0), 1, 1, 1), transformations, padding_mode="border")
    #
    #     return torch.sum(k(model, tx, ty) * w)
    #
    # ys = []
    # zs = []
    #
    # with torch.no_grad():
    #     c = K(x, x).item()
    #
    #     for alpha in tqdm(range(-60, 61)):
    #         x_ = TransformPipeline(SpatialTransformation(rot=(alpha / 30, 0, 0), dst_size=(32, 32)))(x)
    #         ys.append(K(x, x_).item() / c)
    #         zs.append(k(model, T(x), x_).item())
    #
    # plt.plot(ys, label="invariant")
    # plt.plot(zs, label="original")
    # plt.legend()
    # plt.show()


main()
# for p in args.model_path:
#     with torch.no_grad():
#         m = load_model(p)
#         center = F.normalize(m(TransformPipeline(SpatialTransformation(dst_size=(32, 32)))(x)))
#
#         ys = []
#
#         for alpha in range(-60, 61):
#             T = TransformPipeline(SpatialTransformation(rot=(alpha / 30, 0, 0), dst_size=(32, 32)))
#             t = F.normalize(m(T(x)))
#             ys.append(torch.mean(torch.sum(center * t, dim=1)).item())
#
#     ys = np.asarray(ys)
#     ws = 51
#     plt.plot(ys, label=p)
#     plt.plot(np.convolve(np.pad(ys, (ws//2, ws//2), mode="edge"), np.ones((ws,))/ws, mode="valid"))
#
# plt.legend()
# plt.show()
