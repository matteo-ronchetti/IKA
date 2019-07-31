import torch
import torch.nn as nn
from ika import IKA, distance_matrix


class RBFModel(nn.Module):
    def __init__(self, filters, sigma):
        super().__init__()

        self.filters = nn.Parameter(filters)
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(-distance_matrix(x, self.filters, squared=True) / (2 * self.sigma ** 2))


def main():
    X = torch.randn((1000, 32))

    D = distance_matrix(X, X)

    sigma, _ = torch.kthvalue(D.view(-1), int(0.1 * D.size(0) ** 2), dim=-1)
    sigma = sigma.item()

    filters = torch.randn((64, 32))
    b = RBFModel(filters, sigma)

    ika = IKA(b)

    G = torch.exp(-0.5 * (D / sigma) ** 2)
    ika.compute_linear_layer(X, G)

    with torch.no_grad():
        y = ika(X)
        G_ = y @ y.t()

        # error = torch.norm(G - G_) / torch.norm(G)
        # print(error.item())
        loss = torch.mean((G - G_) ** 2)
        print(loss.item())

    optimizer = torch.optim.Adam(b.parameters(), lr=1e-2)

    for i in range(50):
        X = torch.randn((3000, 32))
        G = torch.exp(-0.5 * (distance_matrix(X, X) / sigma) ** 2)

        y = ika(X)
        G_ = y @ y.t()

        loss = torch.mean((G - G_) ** 2)
        print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ika.compute_linear_layer(X, G)


main()
