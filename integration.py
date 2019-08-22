import numpy as np
import scipy.optimize
from scipy.spatial import distance_matrix
from ika import distance_matrix as torch_distance_matrix
from scipy.sparse.linalg import cg
from tqdm import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

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

    def estimate(self, grid, f):
        G = np.exp(-0.5 * distance_matrix(grid, grid) ** 2 / self.kernel_sigma)
        c = np.exp(-(0.5 * np.linalg.norm(grid, axis=1) ** 2) / (self.kernel_sigma + self.sigma))

        c *= (self.kernel_sigma / (self.kernel_sigma + self.sigma)) ** (grid.shape[1] / 2)

        w, _ = cg(G, c)

        return float(np.inner(w, f(grid)))


delta = 1.0
d = 3
estimator = GaussianExpectedValueEstimator(0.5, 1.0)

grid = estimator.compute_grid(d, 12, [0.2, 0.3, 0.7], optimize_iter=0)
f = lambda x: np.exp(-0.5 * np.linalg.norm(x - delta, axis=1) ** 2)

s = np.random.randn(1000, d)
xx = s #np.arange(-3, 3, 0.05).reshape(-1, 1)
g = torch.FloatTensor(grid)
x = torch.FloatTensor(xx)
y = torch.FloatTensor(f(grid).reshape(-1, 1))
y2 = torch.FloatTensor(f(x))
kernel_sigma = Variable(torch.FloatTensor(1), requires_grad=True)
kernel_sigma.data[0] = 0.2
optimizer = torch.optim.Adam([kernel_sigma])

# with torch.no_grad():
for i in range(1000):
# for kernel_sigma in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    G = torch.exp(-0.5 * torch_distance_matrix(g, g, squared=True) / kernel_sigma)
    G2 = torch.exp(-0.5 * torch_distance_matrix(x, g, squared=True) / kernel_sigma)

    w, _ = torch.solve(y, G)
    y_ = (G2 @ w).view(-1)

    loss = torch.mean((y2 - y_) ** 2)

    # print(kernel_sigma, loss.item())
    if (i + 1) % 100 == 0:
        print(loss.item())
        print(kernel_sigma.item())
        print(".....")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(kernel_sigma.item())

estimator.kernel_sigma = 0.6  # kernel_sigma.item()
# grid = estimator.compute_grid(d, 12, [0.2, 0.3, 0.4, 0.7], optimize_iter=1000)

# print(2.2 ** (-(3.0 / 2)))
print(0.5 ** (d / 2) * np.exp(-0.25 * delta ** 2 * d))
print(grid.shape[0], estimator.estimate(grid, f))

s = 100
samples = np.random.multivariate_normal(np.zeros(2), np.eye(2), (s, d))
print(s, np.mean(f(samples)))

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
