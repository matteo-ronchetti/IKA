import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


from architectures.layers import Exp
from architectures.hardnet import HardNet
from ika import IKA

from torchvision.datasets import PhotoTour
from torch.utils.data import DataLoader
from transformation import TransformPipeline, SpatialTransformation


def error_rate_at_recall(distances, labels, recall=0.95):
    labels = labels[np.argsort(distances)]
    # Sliding threshold: get first index where recall >= recall_point.
    # This is the index where the number of elements with label==1 below the threshold reaches a fraction of
    # 'recall_point' of the total number of elements with label==1.
    # (np.argmax returns the first occurrence of a '1' in a bool array).
    threshold_index = np.argmax(np.cumsum(labels) >= recall * np.sum(labels))

    FP = np.sum(labels[:threshold_index] == 0)  # Below threshold (i.e., labelled positive), but should be negative
    TN = np.sum(labels[threshold_index:] == 0)  # Above threshold (i.e., labelled negative), and should be negative
    print(FP, TN)
    return 100 * float(FP) / float(FP + TN)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="liberty")
    parser.add_argument("--model", default="hardnet")
    parser.add_argument("--model-path", default="models/checkpoint_liberty_no_aug.pth")
    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f"Found GPU {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    else:
        device = torch.device("cpu")

    if args.model == "hardnet":
        model = HardNet.from_file(args.model_path, device)
    elif args.model == "hardnet+ika":
        ika_features = nn.Sequential(
            HardNet(),
            #nn.Linear(128, 128),
            nn.Linear(128, 1024),
            Exp()
        )

        data = torch.load(args.model_path, map_location=device)
        ika_features.load_state_dict(data["features"])
        ika_features.eval()

        model = IKA(ika_features)
        model.linear = data["ika"]
    else:
        raise Exception()

    model.eval()
    model.to(device)

    dataset = PhotoTour("data/phototour", args.dataset, download=True, train=False)

    T = TransformPipeline(
        SpatialTransformation(dst_size=(32, 32))
    )

    dataloader = DataLoader(dataset, 2*1024, shuffle=False, pin_memory=True, drop_last=False)

    distances = []
    labels = []
    with torch.no_grad():
        for data_a, data_p, label in tqdm(dataloader):
            data_a, data_p = data_a.unsqueeze(1).float().to(device) / 255, data_p.unsqueeze(1).float().to(device) / 255

            out_a = F.normalize(model(T(data_a)), dim=1)
            out_p = F.normalize(model(T(data_p)), dim=1)
            dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
            distances.append(dists.data.cpu().numpy().reshape(-1, 1))
            ll = label.data.cpu().numpy().reshape(-1, 1)
            labels.append(ll)

    labels = np.vstack(labels).reshape(-1)
    distances = np.vstack(distances).reshape(-1)
    print(labels.shape, distances.shape)

    print("Error rate at 95% recall", error_rate_at_recall(distances, labels, 0.95))


main()
