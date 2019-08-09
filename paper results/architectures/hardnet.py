import torch
import torch.nn as nn

from .layers import Flatten, L2Norm, Exp, NormalizeImages


class HardNet(nn.Module):
    """HardNet model definition
    """

    def __init__(self, _stride=2):
        super(HardNet, self).__init__()

        self.normalize = NormalizeImages()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=_stride, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=_stride, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
            Flatten(),
            L2Norm()
        )

    def forward(self, x):
        x = self.normalize(x)
        return self.features(x)

    @staticmethod
    def from_file(path, device):
        hardnet = HardNet().to(device)
        checkpoint = torch.load(path, map_location=device)
        hardnet.load_state_dict(checkpoint['state_dict'])
        return hardnet

# class HardNetKernel(nn.Module):
#     def __init__(self, _stride=2):
#         super().__init__()
#
#         self.features = nn.Sequential(
#             NormalizeImages(),
#             nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(32, affine=False),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(32, affine=False),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=_stride, padding=1, bias=False),
#             nn.BatchNorm2d(64, affine=False),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(64, affine=False),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=_stride, padding=1, bias=False),
#             nn.BatchNorm2d(128, affine=False),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(128, affine=False),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Conv2d(128, 128, kernel_size=8, bias=False),
#             nn.BatchNorm2d(128, affine=False),
#             Flatten(),
#             L2Norm(),
#             nn.Linear(128, 1024),
#             Exp()
#         )
#
#     def forward(self, x):
#         return self.features(x)
