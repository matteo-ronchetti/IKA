import numpy as np
import scipy.optimize
from scipy.spatial import distance_matrix
from ika import distance_matrix as torch_distance_matrix
from ika import IKA
from scipy.sparse.linalg import cg
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from architectures.layers import Exp
from architectures.hardnet import HardNet
from torchvision.datasets import PhotoTour
from torch.utils.data import DataLoader
from transformation import TransformPipeline, SpatialTransformation

import torch
from torch.autograd import Variable


def spherical_grid(d, n_angles, radiuses):
    if d == 1:
        grid = np.concatenate([np.asarray(radiuses), -np.asarray(radiuses)]).reshape(-1, 1)
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
        for r in radiuses:
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

        def f(k):
            with torch.no_grad():
                G = torch.exp(-0.5 * torch_distance_matrix(k * x, k * x, squared=True) / self.kernel_sigma)
                c = torch.exp(-(0.5 * torch.sum((k * x) ** 2, dim=1)) / (self.sigma + self.kernel_sigma)).view(-1, 1)

                y, _ = torch.solve(c, G)
                return (-c.t() @ y).item()

        k = scipy.optimize.minimize_scalar(f, method="bounded", bounds=[0.1, 4.0]).x
        print(k)

        x.data *= k
        optimizer.zero_grad()

        for i in tqdm(range(optimize_iter)):
            G = torch.exp(-0.5 * torch_distance_matrix(x, x, squared=True) / self.kernel_sigma)
            c = torch.exp(-(0.5 * torch.sum(x ** 2, dim=1)) / (self.sigma + self.kernel_sigma)).view(-1, 1)

            y, _ = torch.solve(c, G)
            loss = - c.t() @ y
            if (i + 1) % 100 == 0:
                print(loss.item())

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


ika_features = nn.Sequential(
    HardNet(),
    nn.Linear(128, 128),
    nn.Linear(128, 1024 * 4),
    Exp()
)

if torch.cuda.is_available():
    print(f"Found GPU {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')
else:
    device = torch.device("cpu")

data = torch.load("models/model_4096.pth", map_location=device)
ika_features.load_state_dict(data["features"])
ika_features.eval()

model = IKA(ika_features)
model.linear = data["ika"]

model.eval()
model = model.to(device)

dataset = PhotoTour("data/phototour", "liberty", download=True, train=True)
dataloader = DataLoader(dataset, 1, shuffle=False, pin_memory=True, drop_last=False)

estimator = GaussianExpectedValueEstimator(0.25, 1.0)
train_points = estimator.compute_grid(3, 12, [0.2, 0.3, 0.4, 0.7], optimize_iter=3000)
print("Grid has", train_points.shape[0], "points")

train_grid = torch.cat(
    [SpatialTransformation.compute_grid(32, 32, translation=(p[1] / 50, p[2] / 50), rot=(p[0] / 50, 0, 0),
                                        scale=1).unsqueeze(0) for p in
     train_points]).to(device)

# test_points = np.random.randn(200, 3)
# test_grid = torch.cat(
#     [SpatialTransformation.compute_grid(32, 32, translation=(p[1] / 50, p[2] / 50), rot=(p[0] / 50, 0, 0),
#                                         scale=1).unsqueeze(0) for p in
#      test_points])
#
# # find an optimal kernel sigma
# # TODO properly minimize f
# D_train = torch.FloatTensor(distance_matrix(train_points, train_points) ** 2)
# D_test = torch.FloatTensor(distance_matrix(test_points, train_points) ** 2)
# x = next(iter(dataloader)).unsqueeze(1).float() / 255
#
#
# def f(kernel_sigma):
#     with torch.no_grad():
#         y_train = model(F.grid_sample(x.repeat(train_grid.size(0), 1, 1, 1), train_grid, padding_mode="border"))
#         y_test = model(F.grid_sample(x.repeat(test_grid.size(0), 1, 1, 1), test_grid, padding_mode="border"))
#
#         G = torch.exp(-0.5 * D_train / kernel_sigma)
#         G2 = torch.exp(-0.5 * D_test / kernel_sigma)
#
#         w, _ = torch.solve(y_train, G)
#         y_ = (G2 @ w)
#
#         loss = torch.mean((y_test - y_) ** 2).item()
#         print(kernel_sigma, loss)
#         return loss
#
#
# # k = scipy.optimize.minimize_scalar(f, [0.1, 2.0], bounds=[0.0, 4.0])
# # print(k)

w = torch.FloatTensor(estimator.quadrature_weights(train_points)).to(device)

Y = torch.empty((40000, 4096)).to(device)
i = 0

for x in tqdm(dataloader):
    x = x.unsqueeze(1).float().to(device) / 255
    with torch.no_grad():
        Y[i] = w @ model(F.grid_sample(x.repeat(train_grid.size(0), 1, 1, 1), train_grid, padding_mode="border"))
        i += 1
        if i == 40000:
            break

np.save("integrated.npy", Y.cpu().data.numpy())


# x = next(iter(dataloader)).unsqueeze(1).float() / 255
# # for i in range(100):
# for kernel_sigma in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0]:
#     with torch.no_grad():
#         y_train = model(F.grid_sample(x.repeat(train_grid.size(0), 1, 1, 1), train_grid, padding_mode="border"))
#         y_test = model(F.grid_sample(x.repeat(test_grid.size(0), 1, 1, 1), test_grid, padding_mode="border"))
#
#         G = torch.exp(-0.5 * D_train / kernel_sigma)
#         G2 = torch.exp(-0.5 * D_test / kernel_sigma)
#
#         w, _ = torch.solve(y_train, G)
#         y_ = (G2 @ w)
#
#         loss = torch.mean((y_test - y_) ** 2)
#
#         print(kernel_sigma, loss.item())
#     # if (i + 1) % 10 == 0:
#     #     print(loss.item())
#     #     print(kernel_sigma.item())
#     #     print(".....")
#     #
#     # optimizer.zero_grad()
#     # loss.backward()
#     # optimizer.step()
#     #
#     # i += 1

#
# x = next(iter(dataloader)).unsqueeze(1).float() / 255
#
# grid = torch.cat(
#     [SpatialTransformation.compute_grid(32, 32, translation=(0, 0, 0), rot=(a / 50, 0, 0), scale=1).unsqueeze(0) for a in
#      range(-20, 20)])
#
# print(grid.size())
#
# alphas = []
# ys = []
# x = x.repeat(grid.size(0), 1, 1, 1)
#
# y = model(F.grid_sample(x, grid, padding_mode="border"))
# print(y.size())
#
# plt.plot(range(-20, 20), y.data.cpu().numpy()[:, 4000])
# plt.show()
# y = model(T(x))
# alphas.append(alpha)
# ys.append(y[0, 0])

# plt.plot(alphas, ys)
# plt.show()

# delta = 1.0
# d = 3
# estimator = GaussianExpectedValueEstimator(0.5, 1.0)
#
# grid = estimator.compute_grid(d, 12, [0.2, 0.3, 0.7], optimize_iter=0)
# f = lambda x: np.exp(-0.5 * np.linalg.norm(x - delta, axis=1) ** 2)
#
# s = np.random.randn(1000, d)
# xx = s #np.arange(-3, 3, 0.05).reshape(-1, 1)
# g = torch.FloatTensor(grid)
# x = torch.FloatTensor(xx)
# y = torch.FloatTensor(f(grid).reshape(-1, 1))
# y2 = torch.FloatTensor(f(x))
# kernel_sigma = Variable(torch.FloatTensor(1), requires_grad=True)
# kernel_sigma.data[0] = 0.2
# optimizer = torch.optim.Adam([kernel_sigma])
#
# # with torch.no_grad():
# for i in range(1000):
# # for kernel_sigma in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#     G = torch.exp(-0.5 * torch_distance_matrix(g, g, squared=True) / kernel_sigma)
#     G2 = torch.exp(-0.5 * torch_distance_matrix(x, g, squared=True) / kernel_sigma)
#
#     w, _ = torch.solve(y, G)
#     y_ = (G2 @ w).view(-1)
#
#     loss = torch.mean((y2 - y_) ** 2)
#
#     # print(kernel_sigma, loss.item())
#     if (i + 1) % 100 == 0:
#         print(loss.item())
#         print(kernel_sigma.item())
#         print(".....")
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
# print(kernel_sigma.item())
#
# estimator.kernel_sigma = 0.6  # kernel_sigma.item()
# # grid = estimator.compute_grid(d, 12, [0.2, 0.3, 0.4, 0.7], optimize_iter=1000)
#
# # print(2.2 ** (-(3.0 / 2)))
# print(0.5 ** (d / 2) * np.exp(-0.25 * delta ** 2 * d))
# print(grid.shape[0], estimator.estimate(grid, f))
#
# s = 100
# samples = np.random.multivariate_normal(np.zeros(2), np.eye(2), (s, d))
# print(s, np.mean(f(samples)))

# plt.plot(xx, f(xx))
# plt.scatter(grid.reshape(-1), f(grid))
# # plt.scatter(xx, y2.data.numpy())
# plt.plot(xx, y_.data.numpy())
# plt.show()
# print(np.min(grid[:, 0]), np.max(grid[:, 0]))

#
# # print(1 + 4*8**5)
# # print(grid.shape)
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2])
# # plt.show()
# # sdsdsd()
# f = lambda xx: 1.0 / (1.0 + 3 * xx ** 2)
# p = lambda xx: np.exp(-0.5 * xx ** 2) / np.sqrt(2 * np.pi)
# k = lambda xx, yy: np.exp(-0.5 * distance_matrix(xx, yy) ** 2 / 0.1)
#
# # x = np.arange(-20, 20, 0.000001)
# # print(x.shape[0], np.mean(f(x) * p(x)) * 40)
# print("Correct", 0.481872302)
#
# x = Variable(torch.FloatTensor(grid), requires_grad=True)
# optimizer = torch.optim.Adam([x])
#
# for i in range(3000):
#     G = torch.exp(-0.5 * torch_distance_matrix(x, x, squared=True) / 0.1)
#     c = torch.exp(-(0.5 * torch.sum(x ** 2, dim=1)) / (1 + 0.1)).view(-1, 1)
#
#     # L = torch.cholesky(G)
#     # print(torch.solve(c, G))
#     y, _ = torch.solve(c, G)
#     loss = - c.t() @ y
#     if (i + 1) % 100 == 0:
#         print(loss.item())
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     # print(x.grad)
#
# # # # x = np.concatenate([np.arange(-5, 5, 1), np.random.randn(10)]).reshape(-1, 1)
# x = x.data.numpy()  # np.random.randn(10).reshape(-1, 1)
# # x = grid
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x[:, 0], x[:, 1], x[:, 2])
# plt.show()

# print(x)
# G = k(x, x)
# c = np.exp(-(0.5 * x ** 2) / (1 + 0.1)) * np.sqrt(0.1 / (1.0 + 0.1))
# v, _ = cg(G, f(x))
# w, _ = cg(G, c)
# m = lambda xx: k(xx.reshape(-1, 1), x) @ v
#
# print(x.shape[0], np.inner(w, f(x.reshape(-1))))
# # print(x.shape[0], np.mean(f(x)*p(x)) * 10)
#
#
# xx = np.arange(-5, 5, 0.05)
# plt.plot(xx, f(xx))
# plt.plot(xx, m(xx))
# plt.scatter(x[:, 0], x[:, 1])
# plt.show()

# for s in [10, 100, 1000, 10000, 20000, 50000, 100000, 1000000]:
#     samples = np.random.randn(s)
#     print(s, np.mean(f(samples)))
#
# for s in [0.5, 0.3, 0.03, 0.003, 0.0003]:
#     x = np.arange(-3, 3, s)
#     print(x.shape[0], np.mean(f(x)*p(x)) * 6)

# plt.plot(x, f)
# plt.plot(x, p)
# plt.hist(samples, weights=np.ones(samples.shape[0]) / samples.shape[0])
# plt.show()
