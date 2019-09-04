import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from architectures.layers import Exp
from architectures.hardnet import HardNet
from ika import IKA
import matplotlib.pyplot as plt

from torchvision.datasets import PhotoTour
from torch.utils.data import DataLoader
from transformation import TransformPipeline, SpatialTransformation


def load_model(path):
    ika_features = nn.Sequential(
        HardNet(),
        nn.Linear(128, 1024),
        Exp()
    )

    data = torch.load(path, map_location=device)
    ika_features.load_state_dict(data["features"])
    ika_features.eval()

    model = IKA(ika_features)
    model.linear = data["ika"]
    return model


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="liberty")
parser.add_argument("--model-path", nargs='+', default="models/checkpoint_liberty_no_aug.pth")
args = parser.parse_args()

if torch.cuda.is_available():
    print(f"Found GPU {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')
else:
    device = torch.device("cpu")


dataset = PhotoTour("data/phototour", args.dataset, download=True, train=False)
x = dataset.data[:100].unsqueeze(1).float().to(device) / 255


for p in args.model_path:
    with torch.no_grad():
        m = load_model(p)
        center = F.normalize(m(TransformPipeline(SpatialTransformation(dst_size=(32, 32)))(x)))

        ys = []

        for alpha in range(-60, 61):
            T = TransformPipeline(SpatialTransformation(rot=(alpha / 30, 0, 0), dst_size=(32, 32)))
            t = F.normalize(m(T(x)))
            ys.append(torch.mean(torch.sum(center * t, dim=1)).item())

    plt.plot(ys, label=p)

plt.legend()
plt.show()
