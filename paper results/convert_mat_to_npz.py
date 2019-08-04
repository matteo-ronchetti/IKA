import numpy as np
from scipy.io import loadmat

X = loadmat("data/patch_retrieval_Rome_train.mat")["patches"]
y = loadmat("data/patch_retrieval_Rome_train_labels.mat")
y = np.concatenate((y["labels_gt"].reshape(-1), y["labels"].reshape(-1)))

X_test = loadmat("data/patch_retrieval_Rome_test.mat")["patches"]
y_test = loadmat("data/patch_retrieval_Rome_test_labels.mat")
y_test = np.concatenate((y_test["labels_gt"].reshape(-1), y_test["labels"].reshape(-1)))

np.savez("data/rome_patches.npz", X=X, y=y, X_test=X_test, y_test=y_test)
