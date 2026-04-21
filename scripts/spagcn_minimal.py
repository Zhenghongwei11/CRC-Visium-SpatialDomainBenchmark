from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """Dense-friendly GCN layer (SpaGCN-style)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = Parameter(torch.empty(self.in_features, self.out_features))
        self.bias = Parameter(torch.empty(self.out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.mm(x, self.weight)
        if adj.is_sparse:
            out = torch.spmm(adj, support)
        else:
            out = torch.mm(adj, support)
        if self.bias is not None:
            out = out + self.bias
        return out


class SimpleGCDEC(nn.Module):
    """GC-DEC core used by SpaGCN (with k-means init only)."""

    def __init__(self, nfeat: int, nhid: int, alpha: float = 0.2) -> None:
        super().__init__()
        self.gc = GraphConvolution(int(nfeat), int(nhid))
        self.nhid = int(nhid)
        self.alpha = float(alpha)
        self.mu: Parameter | None = None
        self.n_clusters: int | None = None

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.gc(x, adj)
        if self.mu is None:
            raise RuntimeError("Cluster centers (mu) not initialised")
        q = 1.0 / ((1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha) + 1e-8)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return z, q

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    @staticmethod
    def kld(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sum(p * torch.log(p / (q + 1e-6)), dim=1))

    def fit(
        self,
        x: np.ndarray,
        adj: np.ndarray,
        *,
        lr: float,
        max_epochs: int,
        weight_decay: float,
        init_spa: bool,
        n_clusters: int,
        tol: float,
        update_interval: int = 3,
    ) -> np.ndarray:
        x_t = torch.as_tensor(x, dtype=torch.float32)
        adj_t = torch.as_tensor(adj, dtype=torch.float32)

        optimizer = optim.Adam(self.parameters(), lr=float(lr), weight_decay=float(weight_decay))

        with torch.no_grad():
            features = self.gc(x_t, adj_t).cpu().numpy() if init_spa else x

        km = KMeans(int(n_clusters), n_init=20, random_state=0)
        y_pred = km.fit_predict(features)
        self.n_clusters = int(n_clusters)
        self.mu = Parameter(torch.empty(self.n_clusters, self.nhid))

        centers = np.zeros((self.n_clusters, features.shape[1]), dtype=np.float32)
        for c in range(self.n_clusters):
            mask = y_pred == c
            if not np.any(mask):
                centers[c] = features.mean(axis=0)
            else:
                centers[c] = features[mask].mean(axis=0)
        self.mu.data.copy_(torch.as_tensor(centers, dtype=torch.float32))

        y_last = y_pred.copy()
        p = None
        self.train()
        for epoch in range(int(max_epochs)):
            if epoch % int(update_interval) == 0:
                _, q = self.forward(x_t, adj_t)
                p = self.target_distribution(q).detach()

            if p is None:
                raise RuntimeError("Target distribution not initialised")

            optimizer.zero_grad(set_to_none=True)
            _, q = self.forward(x_t, adj_t)
            loss = self.kld(p, q)
            loss.backward()
            optimizer.step()

            y_pred = torch.argmax(q, dim=1).detach().cpu().numpy()
            if epoch > 0 and (epoch - 1) % int(update_interval) == 0:
                delta = float(np.mean(y_pred != y_last))
                if delta < float(tol):
                    break
            y_last = y_pred

        return y_pred

    def predict(self, x: np.ndarray, adj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.eval()
        x_t = torch.as_tensor(x, dtype=torch.float32)
        adj_t = torch.as_tensor(adj, dtype=torch.float32)
        with torch.no_grad():
            z, q = self.forward(x_t, adj_t)
        return z.cpu().numpy(), q.cpu().numpy()


@dataclass
class SpaGCNMinimalConfig:
    num_pcs: int = 50
    lr: float = 0.05
    max_epochs: int = 200
    weight_decay: float = 0.0
    init_spa: bool = True
    tol: float = 5e-3


class SpaGCNMinimal:
    """Minimal SpaGCN-style runner (GC-DEC), without scanpy/numba/histology IO."""

    def __init__(self, config: SpaGCNMinimalConfig | None = None) -> None:
        self.l: float | None = None
        self.config = config or SpaGCNMinimalConfig()
        self.model: SimpleGCDEC | None = None
        self.embed: np.ndarray | None = None
        self.adj_affinity: np.ndarray | None = None

    def set_l(self, l: float) -> None:
        self.l = float(l)

    def train(self, x: np.ndarray, adj_dist: np.ndarray, *, n_clusters: int) -> None:
        if self.l is None:
            raise RuntimeError("SpaGCNMinimal requires l to be set before training")
        if x.shape[0] != adj_dist.shape[0] or adj_dist.shape[0] != adj_dist.shape[1]:
            raise ValueError("Adjacency shape mismatch")

        n_components = min(int(self.config.num_pcs), x.shape[1], max(2, x.shape[0] - 1))
        pca = PCA(n_components=n_components, random_state=0)
        embed = pca.fit_transform(x)

        l_val = float(self.l)
        adj_aff = np.exp(-1.0 * (adj_dist**2) / (2.0 * (l_val**2))).astype(np.float32, copy=False)

        model = SimpleGCDEC(embed.shape[1], embed.shape[1], alpha=0.2)
        _ = model.fit(
            embed,
            adj_aff,
            lr=self.config.lr,
            max_epochs=self.config.max_epochs,
            weight_decay=self.config.weight_decay,
            init_spa=self.config.init_spa,
            n_clusters=int(n_clusters),
            tol=self.config.tol,
        )

        self.model = model
        self.embed = embed.astype(np.float32, copy=False)
        self.adj_affinity = adj_aff

    def predict(self) -> np.ndarray:
        if self.model is None or self.embed is None or self.adj_affinity is None:
            raise RuntimeError("Model not trained")
        _, q = self.model.predict(self.embed, self.adj_affinity)
        return np.argmax(q, axis=1).astype(int)
