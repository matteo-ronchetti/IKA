import torch
from torchvision.datasets import PhotoTour
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformation import TransformPipeline, SpatialTransformation
from architectures.hardnet import HardNet, HardNetKernel
from utils.data import feed_model
from utils.kmeans import kmeans
from ika import IKA


def main():
    if torch.cuda.is_available():
        print(f"Found GPU {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    else:
        device = torch.device("cpu")

    pt = PhotoTour("data/phototour", "liberty", download=True)

    model = HardNet().to(device)
    model.eval()
    checkpoint = torch.load("models/HardNet++.pth", map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    T = TransformPipeline(
        SpatialTransformation(dst_size=(32, 32))
    )

    Y = feed_model(pt, lambda x: model(T(x)), device, 128)

    filters = kmeans(Y, 1024, n_iter=30, n_init=5, spherical=True)

main()
