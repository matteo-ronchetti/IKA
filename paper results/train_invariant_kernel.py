import argparse
import os
import gc
import scipy
import scipy.linalg
import torch
import numpy as np
import copy
from torchvision.datasets import PhotoTour
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split
from utils.data import feed_model
import torchvision
# from matplotlib import pyplot as plt

from tqdm import tqdm
from architectures.hardnet import HardNet
from architectures.layers import Exp
from transformation import TransformPipeline, SpatialTransformation, Contrast, Greyscale
from transformation.random import TruncatedNormal
from utils.kmeans import kmeans
from ika import IKA


def random_transform(s=0.02):
    tn = TruncatedNormal()

    return TransformPipeline(
        SpatialTransformation(translation=(tn(0, s), tn(0, s)), rot=(tn(0, s), tn(0, s), tn(0, s)),
                              scale=tn(1, s), dst_size=(32, 32)),
        # Contrast(tn(1, 0.1)),
        # Greyscale()
    )


def load_npz(path, *names):
    obj = np.load(path)
    return (obj[n] for n in names)


class BatchGenerator:
    def __init__(self, batch_size, X, device):
        self.batch_size = batch_size
        self.X = X

        self.s = 0
        self.size = self.X.shape[0]

        self.device = device

        self.p = None
        self.shuffle()

    def shuffle(self):
        self.s = 0
        self.p = np.random.permutation(self.size)

    def next_batch(self):
        if self.s + self.batch_size > self.size:
            self.shuffle()

        x = self.X[self.p[self.s: self.s + self.batch_size]]
        self.s += self.batch_size

        return (x.float() / 255).to(self.device)


def compute_gramian(X, Y, kernel, model, transformations):
    X = X.float() / 255
    Y = Y.float() / 255
    print("X", torch.min(X), torch.mean(X), torch.max(X))
    print("Y", torch.min(Y), torch.mean(Y), torch.max(Y))

    pb = tqdm(total=len(transformations) ** 2)

    with torch.no_grad():
        G = torch.zeros((X.size(0), X.size(0)), dtype=torch.float32).to(X.device)

        # TODO better mean computation
        for tx in transformations:
            fx = model(tx(X))
            for ty in transformations:
                fy = model(ty(Y))
                G += kernel(fx, fy)
                pb.update(1)
        pb.close()

        G /= len(transformations) ** 2

    return G


def get_dataset_and_default_transform(name: str):
    if name.startswith("rome"):
        if name == "rome-test":
            X = np.load("data/rome_patches.npz")["X_test"][1000:]
        else:
            X = np.load("data/rome_patches.npz")["X"][1000:]

        # dataset = TensorDataset(X)

        T = TransformPipeline(
            SpatialTransformation(dst_size=(32, 32)),
            Greyscale()
        )
    elif name == "liberty":
        dataset = PhotoTour("data/phototour", "liberty", download=True)

        X = dataset.data.unsqueeze(1)
        T = TransformPipeline(
            SpatialTransformation(dst_size=(32, 32))
        )
    else:
        raise Exception(f"Unknown dataset '{name}'")

    return X, T


def RBF(sigma):
    def k(x, y):
        return torch.exp((x @ y.t() - 1.0) / (sigma ** 2))

    return k


# noinspection PyArgumentList
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="liberty")
    parser.add_argument("--factor", default="")
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--precomputed", default=None)
    parser.add_argument("--functions", default=1024, type=int)
    parser.add_argument("--n-transformations", default=20, type=int)
    parser.add_argument("--gram-size", default=10000, type=int)
    parser.add_argument("--sigma", default=1.0, type=float)

    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--iterations", default=20, type=int)
    parser.add_argument("--iter-size", default=100, type=int)
    parser.add_argument("--accumulation-steps", default=10, type=int)

    parser.add_argument("--hardnet", default="models/checkpoint_liberty_no_aug.pth")
    parser.add_argument("--output", default="factor_model.pth")
    parser.add_argument("--filters", default="filters.npy")
    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f"Found GPU {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    else:
        device = torch.device("cpu")

    # load hardnet
    hardnet = HardNet.from_file(args.hardnet, device)
    # hardnet.eval()

    X, T = get_dataset_and_default_transform(args.dataset)

    if args.factor:
        with torch.no_grad():
            phi = torch.FloatTensor(np.load(args.factor)).to(device)

            if os.path.exists(args.filters):
                filters = np.load(args.filters)
            else:
                print("Feeding dataset through HardNet...")
                features = feed_model(X, lambda x: hardnet(T(x)), device, 128)

                print("Clustering features...")
                filters = kmeans(features, args.functions, n_iter=30, n_init=10, spherical=True)
                np.save(args.filters, filters)

            sigma = args.sigma
            print("Sigma", sigma)

            # create ika features
            W = filters / sigma ** 2
            bias = -np.ones(W.shape[0], dtype=np.float32) / (sigma ** 2)

            print("Feeding dataset through HardNet...")

            # create b function as hardnet + RBF layer
            ika_features = nn.Sequential(
                HardNet.from_file(args.hardnet, device),
                nn.Linear(128, args.functions),
                Exp()
            )
            ika_features[-2].weight.data = torch.FloatTensor(W)
            ika_features[-2].bias.data = torch.FloatTensor(bias)
            ika_features = ika_features.to(device)
            ika_features.eval()

            X_train = X[:55000]
            phi_train = phi[:55000]
            X_test = X[55000:]
            phi_test = phi[55000:]

            B = feed_model(X_train, lambda x: ika_features(T(x)), device, 1024)
            Q, R = torch.qr(B)
            R = R.data.cpu().numpy()
            M = phi_train.t() @ Q
            M = (M.t() @ M).data.cpu().numpy()

            d, V = scipy.linalg.eigh(M)
            d = np.sqrt(np.maximum(d, 0))
            V = scipy.linalg.solve_triangular(R, V)

            linear = torch.FloatTensor(V @ np.diag(d)).to(device)

            G = phi_test @ phi_test.t()
            print(G.size())
            model = IKA(ika_features)
            model.linear = linear

            print("Usage", torch.cuda.memory_allocated())

            del B
            del Q
            del M
            del phi
            del X_train
            del phi_train

            print("Usage", torch.cuda.memory_allocated())

            gc.collect()
            print("Usage", torch.cuda.memory_allocated())

            tensors = []
            tot = 0
            for obj in gc.get_objects():
                if torch.is_tensor(obj) and obj.device == device:
                    tot += obj.element_size() * obj.nelement()
                    tensors.append((obj.element_size() * obj.nelement(), obj.size(), obj.device))
            tensors.sort(key=lambda x: x[0], reverse=True)
            for t in tensors:
                print(t)
            print(tot, torch.cuda.memory_allocated())

            x = T(X_test.to(device).float() / 255)
            print(x.size(), torch.cuda.memory_allocated())
            print(model.measure_error(x, None, G))

            torch.save({
                "features": ika_features.state_dict(),
                "ika": linear
            }, args.output)

            return

    # X = X[torch.randperm(X.size(0))]

    # x = X[:10].float() / 255
    # y = hardnet(T(x))
    # kernel = RBF(1.0)
    #
    # tn = TruncatedNormal()
    #
    # seq = []
    # for i in range(1000):
    #     seq.append(torch.mean(compute_gramian(x, x, kernel, hardnet, [
    #         TransformPipeline(SpatialTransformation(translation=(tn(0, 0.02), 0), dst_size=(32, 32)))])).item())
    #
    # seq = np.asarray(seq)
    # mean = np.cumsum(seq) / (1 + np.arange(0, seq.shape[0]))
    # # print(np.cumsum(seq))
    # # print((1 + np.arange(0, seq.shape[0])))
    # plt.plot(mean)
    # plt.show()

    # mean = []
    # var = []
    # for alpha in range(-20, 21):
    #     T = TransformPipeline(SpatialTransformation(translation=(alpha / 10, 0), dst_size=(32, 32)))
    #     G = kernel(hardnet(T(x)), y)
    #     mean.append(torch.mean(G).item())
    #     var.append(torch.var(G).item())

    # mean = np.asarray(mean)
    # var = np.asarray(var)
    # print(var)
    # plt.plot([x for x in range(-20, 21)], mean)
    # plt.fill_between([x for x in range(-20, 21)], mean - var, mean + var,
    #                  color='gray', alpha=0.2)
    # plt.show()
    #
    # # for i in range(3):
    # #     grid_img = torchvision.utils.make_grid(random_transform(0.1)(x), nrow=10)
    # #     plt.figure()
    # #     plt.imshow(grid_img.permute(1, 2, 0))
    # # plt.show()
    # return

    # shuffle X
    # TODO correct removal of val samples when loading precomputed data
    X = X[torch.randperm(X.size(0))]
    X_val = X[-3000:]
    X_val = X_val.to(device)
    X_val_r = X[-6000:-3000]
    X_val_r = X_val_r.to(device)

    X = X[:-6000]

    if args.precomputed:
        # G, gx, X_val, X_val_r, G_val, filters, sigma = load_npz(args.precomputed, "G", "gx", "X_val", "X_val_r", "G_val", "filters", "sigma")
        G, gx, X_val, X_val_r, G_val, filters, sigma = load_npz(args.precomputed, "G", "gx", "X_val", "X_val", "G_val",
                                                                "filters", "sigma")

        G = torch.FloatTensor(G).to(device)
        G_val = torch.FloatTensor(G_val).to(device)
        X_val = torch.FloatTensor(X_val).to(device)
        X_val_r = torch.FloatTensor(X_val_r).to(device)
        gx = torch.FloatTensor(gx).to(device)
        sigma = float(sigma)
        kernel = RBF(sigma)
    else:
        print("Feeding dataset through HardNet...")
        features = feed_model(X, lambda x: hardnet(T(x)), device, 128)

        print("Clustering features...")
        filters = kmeans(features, args.functions, n_iter=50, n_init=10, spherical=True)

        sigma = args.sigma
        print("Sigma", sigma)

        print("Estimating Gramian matrix...")
        if args.n_transformations > 0:
            transformations = [random_transform() for _ in range(args.n_transformations)]
        else:
            transformations = [T]

        kernel = RBF(sigma)
        gx = X[:args.gram_size].float().to(device) / 255
        G = compute_gramian(gx, gx, kernel, hardnet, transformations)
        G_val = compute_gramian(X_val, X_val_r, kernel, hardnet, transformations)

        np.savez("precomputed.npz", G=G.data.cpu().numpy(), gx=gx.data.cpu().numpy(), X_val=X_val.data.cpu().numpy(),
                 X_val_r=X_val_r.data.cpu().numpy(),
                 G_val=G_val.data.cpu().numpy(), filters=filters,
                 sigma=sigma)

    # create ika features
    W = filters / sigma ** 2
    bias = -np.ones(W.shape[0], dtype=np.float32) / (sigma ** 2)

    print("Feeding dataset through HardNet...")
    with torch.no_grad():
        y = feed_model(X[:30000], lambda x: hardnet(T(x)), device, 128)
        mean = torch.mean(y, dim=0)
        # print(mean.size())
        y -= mean[None, :]
        mean = mean.data.cpu().numpy()
        y /= np.sqrt(float(y.size(0)))

        d, V = np.linalg.eigh((y.t() @ y).data.cpu().numpy())
        # print(mean)
        # print(d)

        # d += np.mean(d)
        # d /= d[-1]

        P = V @ np.diag(1 / np.sqrt(d)) @ V.T
        P_inv = V @ np.diag(np.sqrt(d)) @ V.T

        W = W @ P_inv
        bias = bias + W @ mean

    # create b function as hardnet + RBF layer
    ika_features = nn.Sequential(
        HardNet.from_file(args.hardnet, device),
        nn.Linear(128, 128),
        nn.Linear(128, args.functions),
        Exp()
    )
    ika_features[-3].weight.data = torch.FloatTensor(P)
    ika_features[-3].bias.data = torch.FloatTensor(-mean)
    ika_features[-2].weight.data = torch.FloatTensor(W)
    ika_features[-2].bias.data = torch.FloatTensor(bias)
    ika_features = ika_features.to(device)

    # Compute IKA
    ika = IKA(ika_features)
    with torch.no_grad():
        gx = T(gx)
        x_val = T(X_val.float() / 255)
        x_val_r = T(X_val_r.float() / 255)

        print("x_val", torch.min(x_val), torch.mean(x_val), torch.max(x_val))
        print("x_val_r", torch.min(x_val_r), torch.mean(x_val_r), torch.max(x_val_r))

    ika_features.eval()
    ika.compute_linear_layer(gx, G, eps=1e-4)
    print("Error before training", ika.measure_error(x_val, x_val_r, G_val))

    torch.save({
        "features": ika_features.state_dict(),
        "ika": ika.linear
    }, "model.pth")

    # ika_features.train()

    optimizer = torch.optim.Adam(list(ika_features[-2].parameters()), lr=args.lr)

    x_batches = BatchGenerator(args.batch_size, X, device)
    y_batches = BatchGenerator(args.batch_size, X, device)

    optimizer.zero_grad()
    for iteration in range(args.iterations):
        tot_loss = 0
        for i in tqdm(range(args.iter_size)):
            if args.n_transformations > 0:
                tx = random_transform()
                ty = random_transform()
            else:
                tx = T
                ty = T

            xx = x_batches.next_batch()
            y = y_batches.next_batch()
            # print("xx", torch.min(xx).item(), torch.max(xx).item())

            fx = ika(T(xx))
            fy = ika(T(y))

            with torch.no_grad():
                G_ = kernel(hardnet(tx(xx)), hardnet(ty(y)))

            loss = torch.mean((G_ - fx @ fy.t()) ** 2)
            tot_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        ika_features.eval()
        ika.compute_linear_layer(gx, G, eps=1e-4)
        print(f"""Iteration: {iteration + 1}, loss: {tot_loss / args.iter_size}, predicted: {np.sqrt(
            tot_loss / args.iter_size) * 3000}, validation error: {ika.measure_error(
            x_val, x_val_r, G_val)}""")
        # ika_features.train()

    torch.save({
        "features": ika_features.state_dict(),
        "ika": ika.linear
    }, "model.pth")


main()
