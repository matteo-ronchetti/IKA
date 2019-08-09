import argparse
import torch
import numpy as np
import copy
from torchvision.datasets import PhotoTour
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split
from utils.data import feed_model

from tqdm import tqdm
from architectures.hardnet import HardNet
from architectures.layers import Exp
from transformation import TransformPipeline, SpatialTransformation, Contrast, Greyscale
from transformation.random import TruncatedNormal
from utils.kmeans import kmeans
from ika import IKA


def random_transform():
    tn = TruncatedNormal()

    return TransformPipeline(
        SpatialTransformation(translation=(tn(0, 0.02), tn(0, 0.02)), rot=(tn(0, 0.02), tn(0, 0.02), tn(0, 0.02)),
                              scale=tn(1, 0.02), dst_size=(32, 32)),
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


def compute_gramian(X, kernel, model, transformations):
    X = X.float() / 255
    pb = tqdm(total=len(transformations) ** 2)

    with torch.no_grad():
        G = torch.zeros((X.size(0), X.size(0)), dtype=torch.float32).to(X.device)

        # TODO better mean computation
        for tx in transformations:
            fx = model(tx(X))
            for ty in transformations:
                fy = model(ty(X))
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
    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f"Found GPU {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    else:
        device = torch.device("cpu")

    # load hardnet
    hardnet = HardNet.from_file(args.hardnet, device)
    hardnet.eval()

    X, T = get_dataset_and_default_transform(args.dataset)

    # shuffle X
    # TODO correct removal of val samples when loading precomputed data
    X = X[torch.randperm(X.size(0))]
    X_val = X[-3000:]
    X_val = X_val.float().to(device) / 255
    X = X[:-3000]

    if args.precomputed:
        G, gx, X_val, G_val, filters, sigma = load_npz(args.precomputed, "G", "gx", "X_val", "G_val", "filters", "sigma")
        G = torch.FloatTensor(G).to(device)
        G_val = torch.FloatTensor(G_val).to(device)
        X_val = torch.FloatTensor(X_val).to(device)
        gx = torch.FloatTensor(gx).to(device)
        sigma = float(sigma)
        kernel = RBF(sigma)
    else:
        print("Feeding dataset through HardNet...")
        features = feed_model(X, lambda x: hardnet(T(x)), device, 128)

        print("Clustering features...")
        filters = kmeans(features, args.functions, n_iter=30, n_init=5, spherical=True)

        sigma = args.sigma
        print("Sigma", sigma)

        print("Estimating Gramian matrix...")
        if args.n_transformations > 0:
            transformations = [random_transform() for _ in range(args.n_transformations)]
        else:
            transformations = [T]

        kernel = RBF(sigma)
        gx = X[:args.gram_size].float().to(device) / 255
        G = compute_gramian(gx, kernel, hardnet, transformations)
        G_val = compute_gramian(X_val, kernel, hardnet, transformations)

        np.savez("precomputed.npz", G=G.data.cpu().numpy(), gx=gx.data.cpu().numpy(), X_val=X_val.data.cpu().numpy(),
                 G_val=G_val.data.cpu().numpy(), filters=filters,
                 sigma=sigma)

    # create ika features
    W = filters / sigma ** 2
    bias = -np.ones(W.shape[0], dtype=np.float32) / (sigma ** 2)

    # create b function as hardnet + RBF layer
    ika_features = nn.Sequential(
        HardNet.from_file(args.hardnet, device),
        nn.Linear(128, args.functions),
        Exp()
    )
    ika_features[-2].weight.data = torch.FloatTensor(W)
    ika_features[-2].bias.data = torch.FloatTensor(bias)
    ika_features = ika_features.to(device)

    # Compute IKA
    ika = IKA(ika_features)
    with torch.no_grad():
        gx = T(gx)
        x_val = T(X_val)

    ika_features.eval()
    ika.compute_linear_layer(gx, G, eps=1e-4)
    print("Error before training", ika.measure_error(x_val, G_val))

    torch.save({
        "features": ika_features.state_dict(),
        "ika": ika.linear
    }, "model.pth")

    #ika_features.train()

    optimizer = torch.optim.Adam(list(ika_features[-2].parameters()) + list(ika.linear), lr=args.lr)

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

            fx = ika(T(xx))
            fy = ika(T(y))

            with torch.no_grad():
                G_ = kernel(hardnet(tx(xx)), hardnet(ty(y)))

            loss = torch.mean((G_ - fx @ fy.t()) ** 2) / args.accumulation_steps
            tot_loss += loss.item()

            loss.backward()
            if (i + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        ika_features.eval()
        ika.compute_linear_layer(gx, G, eps=1e-4)
        print(f"""Iteration: {iteration + 1}, loss: {tot_loss / args.iter_size}, validation error: {ika.measure_error(
            x_val, G_val)}""")
        #ika_features.train()

    torch.save({
        "features": ika_features.state_dict(),
        "ika": ika.linear
    }, "model.pth")


main()