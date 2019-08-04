import argparse
import torch
import numpy as np
import faiss
import random
import copy
from tqdm import tqdm
from architectures.hardnet import HardNet, HardNetKernel
from transformation import TransformPipeline, SpatialTransformation, Contrast, Greyscale
from transformation.random import TruncatedNormal

from ika import IKA


def kmeans(X, k, n_iter=30, n_init=1, spherical=False, verbose=True, subsample=-1, seed=-1):
    """
    Run kmeans and return centroids
    :param X: data
    :param k: number of clusters
    :param n_iter: number of iterations
    :param n_init: number of times the algorithm will be executed
    :param spherical:
    :param verbose:
    :param subsample: if specified it uses only "subsample" points per centroid
    :param seed:
    :return: centroids
    """
    if seed is None:
        seed = random.seed()
    if subsample == -1:
        subsample = X.shape[0] // k + 1

    km = faiss.Kmeans(X.shape[1], k, niter=n_iter, nredo=n_init, max_points_per_centroid=subsample,
                      min_points_per_centroid=1, verbose=verbose,
                      spherical=spherical, seed=seed)

    if isinstance(X, torch.Tensor):
        x_ptr = X.cpu().numpy()
        # assert X.is_contiguous()
        # assert X.dtype == torch.float32
        # x_ptr = faiss.cast_integer_to_float_ptr(X.storage().data_ptr() + X.storage_offset() * 4)
    else:
        x_ptr = X

    km.train(x_ptr)
    return km.centroids


def random_transform():
    tn = TruncatedNormal()

    return TransformPipeline(
        SpatialTransformation(translation=(tn(0, 0.05), tn(0, 0.05)), rot=(tn(0, 0.05), tn(0, 0.05), tn(0, 0.05)),
                              scale=tn(1, 0.05), dst_size=(32, 32)),
        Contrast(tn(1, 0.2)),
        Greyscale()
    )


def load_npz(path, *names):
    obj = np.load(path)
    return (obj[n] for n in names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/rome_patches.npz")
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--precomputed", default=None)

    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f"Found GPU {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    else:
        device = torch.device("cpu")

    # load training data
    if args.test:
        X = np.load(args.dataset)["X_test"][1000:]
    else:
        X = np.load(args.dataset)["X"][1000:]

    # load hardnet
    model = HardNet().to(device)
    checkpoint = torch.load("models/HardNet++.pth", map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    x = (torch.FloatTensor(X) / 255).to(device)

    T = TransformPipeline(
        SpatialTransformation(dst_size=(32, 32)),
        Greyscale()
    )

    if args.precomputed:
        G, filters, sigma = load_npz(args.precomputed, "G", "filters", "sigma")
        sigma = float(sigma)
    else:
        print("Feeding data through model...")
        with torch.no_grad():
            features = model(T(x)).data.cpu().numpy()

        print("Clustering features...")
        filters = kmeans(features, 1024, n_iter=30, n_init=5, spherical=True)

        D = 2.0 - 2.0 * features @ features.T
        D = np.sqrt(np.maximum(D, 0))
        sigma = np.quantile(D, 0.01)
        print("Sigma", sigma)

        print("Estimating Gramian matrix...")
        transformations = [T] + [random_transform() for i in range(9)]

        pb = tqdm(total=len(transformations) ** 2)
        with torch.no_grad():
            G = torch.zeros((9000, 9000), dtype=torch.float32).to(device)

            # TODO better mean computation
            for tx in transformations:
                fx = model(tx(x))
                for ty in transformations:
                    fy = model(ty(x))
                    G += torch.exp((fx @ fy.t() - 1.0) / sigma ** 2)
                    pb.update(1)
            pb.close()

            G /= len(transformations) ** 2

        np.savez("precomputed.npz", G=G.data.cpu().numpy(), filters=filters, sigma=sigma)

    # create ika features
    W = filters / sigma ** 2
    bias = -np.ones(1024, dtype=np.float32) / (sigma ** 2)

    ika_features = HardNetKernel()
    for i, l in enumerate(model.features):
        ika_features.features[i + 1] = copy.deepcopy(l)
    ika_features.features[-2].weight.data = torch.FloatTensor(W)
    ika_features.features[-2].bias.data = torch.FloatTensor(bias)
    ika_features = ika_features.to(device)

    ika = IKA(ika_features)
    ika.compute_linear_layer(T(x), G)

    with torch.no_grad():
        y = ika(x)
        G_ = y @ y.t()
        loss = torch.mean((G - G_) ** 2)
        print("Error before training", loss.item())

    optimizer = torch.optim.Adam(ika_features.parameters(), lr=1e-4)

    for epoch in range(10):
        pass

main()
