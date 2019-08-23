import math
import torch
import torch.nn.functional as F
import numpy as np
import abc

try:
    import torch_batch_svd
except:
    print("No CUDA batch SVD")


def batch_svd(x):
    U0, S, V0 = torch_batch_svd.batch_svd_forward(x, True, 1e-7, 100)
    k = S.size(1)
    U = U0[:, :, :k]
    V = V0[:, :, :k]

    return U, S, V


def hsv_transform_matrix(h, s, v):
    vsu = v * s * np.cos(h)
    vsw = v * s * np.sin(h)

    V = np.asarray([[.299, .587, .114]] * 3, dtype=np.float32)
    VSU = np.asarray([
        [.701, -.587, -.114],
        [-.299, .413, -.114],
        [-.300, -.588, .886]
    ])

    VSW = np.asarray([
        [.168, .330, -.497],
        [-.328, .035, -.292],
        [1.25, -1.05, .203]
    ])

    return torch.FloatTensor(v * V + vsu * VSU + vsw * VSW)


class Transformation(abc.ABC):
    @abc.abstractmethod
    def compile(self, w, h, device):
        pass

    def __call__(self, x):
        return self.compile(x.size(2), x.size(3), x.device)(x)


class Greyscale(Transformation):
    def __init__(self):
        pass

    def compile(self, w, h, device):
        return lambda x: x.mean(dim=1).unsqueeze(1)


class HSVTransformation(Transformation):
    def __init__(self, h, s=1, v=1):
        self.M = hsv_transform_matrix(h, s, v)

    def compile(self, w, h, device):
        M = self.M.to(device)

        def f(x):
            x = torch.einsum("bcwh,dc->bdwh", x, M)
            return torch.clamp(x, 0, 1)

        return f


class SpatialTransformation(Transformation):
    def __init__(self, translation=(0, 0), rot=(0, 0, 0), scale=1.0, padding_mode="border", dst_size=None):
        self.translation = translation

        if isinstance(rot, float) or isinstance(rot, int):
            self.rot = (0, 0, rot)
        else:
            self.rot = rot

        self.scale = scale
        self.padding_mode = padding_mode
        self.dst_size = dst_size

    @staticmethod
    def compute_grid(H, W, translation, rot, scale):
        grid = torch.empty((H, W, 3))
        grid[:, :, 0] = torch.linspace(-1, 1, W)
        grid[:, :, 1] = torch.linspace(-1, 1, H).unsqueeze(1)
        grid[:, :, 2] = scale

        Rx = np.asarray(
            [[1, 0, 0], [0, np.cos(rot[0]), np.sin(rot[0])], [0, -np.sin(rot[0]), np.cos(rot[0])]])
        Ry = np.asarray(
            [[np.cos(rot[1]), 0, -np.sin(rot[1])], [0, 1, 0], [np.sin(rot[1]), 0, np.cos(rot[1])]])
        Rz = np.asarray(
            [[np.cos(rot[2]), np.sin(rot[2]), 0], [-np.sin(rot[2]), np.cos(rot[2]), 0], [0, 0, 1]])

        R = torch.FloatTensor(Rx @ Ry @ Rz)
        translation = torch.FloatTensor([translation[0], translation[1], 0])

        grid = torch.einsum("hwi,ji->hwj", grid + translation, R)

        grid[:, :, 0] /= grid[:, :, 2]
        grid[:, :, 1] /= grid[:, :, 2]

        t = R @ torch.FloatTensor([0, 0, 1])
        grid -= t

        return grid[:, :, :2].contiguous()

    def compile(self, h, w, device):
        H, W = self.dst_size or (h, w)

        grid = self.compute_grid(H, W, self.translation, self.rot, self.scale).to(device)

        def f(x):
            return F.grid_sample(x, grid.unsqueeze(0).repeat(x.size(0), 1, 1, 1), padding_mode=self.padding_mode)

        return f


class Contrast(Transformation):
    def __init__(self, a=1.0, b=1.0, c=1.0):
        self.d = torch.FloatTensor([a, b, c])

    def compile(self, h, w, device):
        d = self.d.to(device)

        def f(x):
            mean = torch.mean(x, dim=(2, 3), keepdim=True)
            y = x - mean

            cov = torch.einsum("bchw,bdhw->bcd", y, y)

            if device == torch.device('cpu'):
                Ts = []
                for i in range(x.size(0)):
                    _, V = torch.symeig(cov[i], eigenvectors=True)
                    Ts.append(V @ torch.diag(d) @ V.t())
                T = torch.stack(Ts, dim=0)
            else:
                _, s, V = batch_svd(cov)
                T = torch.einsum("bik,k,bjk->bij", V, d, V)

            y = torch.einsum("bcwh,bdc->bdwh", y, T) + mean
            return torch.clamp(y, 0, 1)

        return f


class MotionBlur:
    def __init__(self, radius, direction):
        self.radius = radius
        self.direction = direction

    def compile(self, h, w, device):
        if self.radius == 0:
            return lambda x: x

        blur = torch.zeros(1, 1, 1 + 2 * self.radius, 1 + 2 * self.radius)
        blur[0, 0, 1 + self.radius, self.radius:] = 1.0
        blur = SpatialTransformation(rot=self.direction)(blur)
        blur /= torch.sum(blur)
        blur = blur.contiguous().to(device)

        def f(x):
            x = F.pad(x, [self.radius, self.radius, self.radius, self.radius], mode="replicate")
            return F.conv2d(x, blur.repeat(x.size(1), 1, 1, 1), groups=x.size(1))

        return f


def gaussian_filter_1d(m, sigma):
    """Create 1D Gaussian filter
    """
    filt = torch.arange(-m, m + 1, dtype=torch.float32)
    filt = torch.exp(-filt.pow(2) / (2. * sigma * sigma))
    return filt / torch.sum(filt)


class GaussianBlur(Transformation):
    def __init__(self, sigma, eps=5e-1):
        self.sigma = sigma

        if self.sigma > 0:
            self.radius = self.choose_filter_radius(sigma, eps)

            self.filter = gaussian_filter_1d(self.radius, sigma)

    def compile(self, w, h, device):
        if self.sigma == 0:
            return lambda x: x

        filter = self.filter.to(device)

        def f(x):
            x = F.pad(x, [self.radius, self.radius, self.radius, self.radius], mode="replicate")
            x = F.conv2d(x, filter.repeat(x.size(1), 1, 1, 1), groups=x.size(1))
            return F.conv2d(x, filter.repeat(x.size(1), 1, 1, 1).transpose(2, 3), groups=x.size(1))

        return f

    @staticmethod
    def choose_filter_radius(sigma, eps=5e-1):
        for i in range(2, 50):
            y = math.exp(-(i ** 2) / (2 * sigma ** 2))
            if y <= eps:
                return i - 1

        return 50


class TransformPipeline:
    def __init__(self, *tfs):
        self.tfs = [x for x in tfs if x is not None]

        self.cache = dict()

    def __call__(self, x):
        h = x.size(2)
        w = x.size(3)
        device = x.device

        key = f"{h}_{w}_{device}"

        if key not in self.cache:
            self.cache[key] = [t.compile(h, w, device) for t in self.tfs]

        for t in self.cache[key]:
            x = t(x)

        return x
