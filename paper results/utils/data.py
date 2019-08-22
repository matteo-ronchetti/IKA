import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def feed_model(X, mdl, device, output_size, batch_size=1024 * 2):
    if isinstance(X, torch.Tensor):
        X = TensorDataset(X)

    dl = DataLoader(X, batch_size=batch_size, drop_last=False, pin_memory=True)
    Y = torch.zeros((len(X), output_size), dtype=torch.float, device=device)
    s = 0
    with torch.no_grad():
        for x, in tqdm(dl):
            x = x.to(device).float() / 255

            Y[s: s + x.size(0)] = mdl(x).data.cpu()
            s += x.size(0)

    return Y
