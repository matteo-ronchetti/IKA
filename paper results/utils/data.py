import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def feed_model(X, mdl, device, output_size, batch_size=1024 * 8):
    if isinstance(X, torch.Tensor):
        X = TensorDataset(X)

    dl = DataLoader(X, batch_size=batch_size, drop_last=False, pin_memory=True)
    Y = torch.zeros((X.data.size(0), output_size), dtype=torch.float, device=device)
    s = 0
    with torch.no_grad():
        for x in tqdm(dl):
            x = x.to(device).float().unsqueeze(1) / 255

            Y[s: s + x.size(0)] = mdl.data.cpu()
            s += x.size(0)

    return Y
