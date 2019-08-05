import faiss
import torch
import random


def parse_array(X):
    """
    :param X:
    :return: x_ptr, on_gpu
    """
    if isinstance(X, torch.Tensor):
        if X.device == torch.device("cpu"):
            return X.cpu().numpy(), False
        else:
            assert X.is_contiguous()
            assert X.dtype == torch.float32
            x_ptr = faiss.cast_integer_to_float_ptr(X.storage().data_ptr() + X.storage_offset() * 4)
            return x_ptr, True
    else:
        return X, False


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
    # fill default args
    if seed is None:
        seed = random.seed()
    if subsample == -1:
        subsample = X.shape[0] // k + 1

    # parse input array
    x_ptr, on_gpu = parse_array(X)
    d = X.size(1)

    cp = faiss.ClusteringParameters()
    kwargs = dict(niter=n_iter, nredo=n_init, max_points_per_centroid=subsample,
                  min_points_per_centroid=1, verbose=verbose,
                  spherical=spherical, seed=seed)

    for key, value in kwargs.items():
        # if this raises an exception, it means that it is a non-existent field
        getattr(cp, key)
        setattr(cp, key, value)

    clus = faiss.Clustering(d, k, cp)

    if cp.spherical:
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexFlatL2(d)

    if on_gpu:
        index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)

    clus.train(x_ptr, index)

    return faiss.vector_float_to_array(clus.centroids)

#
# def kmeans(X, k, n_iter=30, n_init=1, spherical=False, verbose=True, subsample=-1, seed=-1):
#     """
#     Run kmeans and return centroids
#     :param X: data
#     :param k: number of clusters
#     :param n_iter: number of iterations
#     :param n_init: number of times the algorithm will be executed
#     :param spherical:
#     :param verbose:
#     :param subsample: if specified it uses only "subsample" points per centroid
#     :param seed:
#     :return: centroids
#     """
#
#     km = faiss.Kmeans(X.shape[1], k, niter=n_iter, nredo=n_init, max_points_per_centroid=subsample,
#                       min_points_per_centroid=1, verbose=verbose,
#                       spherical=spherical, seed=seed)
#
#     km.train(x_ptr)
#     return km.centroids
