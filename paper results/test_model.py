import argparse
import torch
import numpy as np
from architectures.hardnet import HardNet
from transformation import TransformPipeline, SpatialTransformation, Greyscale


def test(x, x_test, y, y_test):
    print(np.min(np.linalg.norm(x, axis=1)), np.max(np.linalg.norm(x, axis=1)))
    # x /= np.linalg.norm(x, axis=1)[:, None]
    # x_test /= np.linalg.norm(x_test, axis=1)[:, None]

    D = -x_test @ x.T  # distance_matrix(x_test, x)

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

    if args.model == "hardnet":
        model = HardNet()
        checkpoint = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])

        T = TransformPipeline(
            SpatialTransformation(dst_size=(32, 32)),
            Greyscale()
        )
    else:
        model = None
        T = TransformPipeline()

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
