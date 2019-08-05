import argparse
import torch
import numpy as np
from architectures.hardnet import HardNet, HardNetKernel
from transformation import TransformPipeline, SpatialTransformation, Greyscale
from ika import IKA

import numba
from numba import prange
import math


@numba.jit(parallel=True, nogil=True, fastmath=True, nopython=True)
def _distance_matrix(D, X, Y):
    for i in prange(D.shape[0]):
        for j in prange(D.shape[1]):
            tmp = 0
            for k in range(X.shape[1]):
                tmp += (X[i, k] - Y[j, k]) ** 2
            D[i, j] = math.sqrt(tmp)


def distance_matrix(X, Y):
    D = np.empty((X.shape[0], Y.shape[0]), dtype=np.float32)
    _distance_matrix(D, X, Y)
    return D


def test(x, x_test, y, y_test):
    print(np.min(np.linalg.norm(x, axis=1)), np.max(np.linalg.norm(x, axis=1)))
    # x /= np.linalg.norm(x, axis=1)[:, None]
    # x_test /= np.linalg.norm(x_test, axis=1)[:, None]

    D = distance_matrix(x_test, x)  # -x_test @ x.T  #

    ranks = np.argsort(D, axis=1)

    print(ranks[0, :10])

    mAP = 0
    for i in range(1000):
        l = y[ranks[i]] == y_test[i]

        recall = np.zeros(l.shape[0] + 1)
        recall[1:] = np.cumsum(l) / np.sum(l)

        precision = np.ones(l.shape[0] + 1)
        precision[1:] = np.cumsum(l) / (np.arange(l.shape[0]) + 1)

        mAP += np.inner((precision[:-1] + precision[1:]) / 2, recall[1:] - recall[:-1])

    return mAP / 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hardnet")
    parser.add_argument("--model-path", default="models/HardNet++.pth")
    parser.add_argument("--dataset", default="data/rome_patches.npz")
    parser.add_argument("--test", action="store_true", default=False)

    args = parser.parse_args()

    if args.test:
        X = np.load(args.dataset)["X_test"]
        y = np.load(args.dataset)["y_test"]
    else:
        X = np.load(args.dataset)["X"]
        y = np.load(args.dataset)["y"]

    T = TransformPipeline(
        SpatialTransformation(dst_size=(32, 32)),
        Greyscale()
    )

    if args.model == "hardnet":
        model = HardNet()
        checkpoint = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
    else:
        features = HardNetKernel()
        data = torch.load("models/model.pth", map_location="cpu")
        features.load_state_dict(data["features"])
        features.eval()

        model = IKA(features)
        model.linear = data["ika"]

    print("Feeding data into model...")
    with torch.no_grad():
        queries = model(T(torch.FloatTensor(X[:1000]) / 255)).data.numpy()

        db = np.empty((9000, queries.shape[1]), dtype=np.float32)
        for i in range(9):
            x = torch.FloatTensor(X[1000 + 1000 * i:i * 1000 + 2000]) / 255
            db[1000 * i:i * 1000 + 1000] = model(T(x)).data.numpy()

    print("Computing MaP...")
    print(test(db, queries, y[1000:], y[:1000]))


main()
