from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def _pca_embed(x: np.ndarray, n_components: int, *, seed: int) -> np.ndarray:
    n_components = max(2, min(int(n_components), x.shape[0] - 1, x.shape[1] - 1))
    model = PCA(n_components=n_components, random_state=int(seed))
    return model.fit_transform(x).astype(np.float32, copy=False)


def _edge_index_from_csr(adj) -> tuple[np.ndarray, np.ndarray]:
    coo = adj.tocoo()
    return coo.row.astype(np.int64, copy=False), coo.col.astype(np.int64, copy=False)


class GATLayerMinimal(nn.Module):
    """
    Minimal single-head GAT layer operating on an edge list.

    Attention is computed as:
      e_ij = LeakyReLU(a_l^T Wh_i + a_r^T Wh_j) for edges (i,j)
      alpha_ij = softmax_j(e_ij) over neighbors of i
      h_i' = sum_j alpha_ij Wh_j

    Implementation uses a sorted edge list and per-node Python loops. This keeps
    dependencies minimal (no torch_scatter/pyg) and is fast enough for kNN graphs.
    """

    def __init__(self, in_dim: int, out_dim: int, *, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.negative_slope = float(negative_slope)
        self.w = nn.Parameter(torch.empty(self.in_dim, self.out_dim))
        self.a_l = nn.Parameter(torch.empty(self.out_dim))
        self.a_r = nn.Parameter(torch.empty(self.out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(max(1, self.out_dim))
        nn.init.uniform_(self.w, -bound, bound)
        nn.init.uniform_(self.a_l, -bound, bound)
        nn.init.uniform_(self.a_r, -bound, bound)

    def forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        # x: (N, Fin); src/dst: (E,)
        h = x @ self.w  # (N, Fout)
        f1 = (h * self.a_l).sum(dim=1)  # (N,)
        f2 = (h * self.a_r).sum(dim=1)  # (N,)

        e = f1[src] + f2[dst]
        e = torch.nn.functional.leaky_relu(e, negative_slope=self.negative_slope)

        n_nodes = int(x.shape[0])

        # Compute max(e) per source node for stable softmax without extra deps.
        max_per_src = torch.full((n_nodes,), -1.0e30, dtype=e.dtype, device=e.device)
        if hasattr(max_per_src, "scatter_reduce_"):
            max_per_src.scatter_reduce_(0, src, e, reduce="amax", include_self=True)
        else:  # pragma: no cover - legacy torch fallback
            for i in range(n_nodes):
                mask = src == i
                if torch.any(mask):
                    max_per_src[i] = torch.max(e[mask])

        exp_e = torch.exp(e - max_per_src[src])
        sum_per_src = torch.zeros((n_nodes,), dtype=e.dtype, device=e.device)
        sum_per_src.index_add_(0, src, exp_e)
        alpha = exp_e / (sum_per_src[src] + 1e-12)

        out = torch.zeros((n_nodes, self.out_dim), dtype=h.dtype, device=h.device)
        out.index_add_(0, src, alpha[:, None] * h[dst])
        return out


class STAGATEAutoEncoderMinimal(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.enc1 = GATLayerMinimal(in_dim, hidden_dim)
        self.enc2 = GATLayerMinimal(hidden_dim, latent_dim)
        self.dec1 = GATLayerMinimal(latent_dim, hidden_dim)
        self.dec2 = GATLayerMinimal(hidden_dim, in_dim)

    def encode(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.enc1(x, src, dst))
        z = self.enc2(h1, src, dst)
        return z

    def decode(self, z: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.dec1(z, src, dst))
        x_hat = self.dec2(h1, src, dst)
        return x_hat

    def forward(self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, src, dst)
        x_hat = self.decode(z, src, dst)
        return z, x_hat


@dataclass
class STAGATEMinimalConfig:
    num_pcs: int = 50
    hidden_dim: int = 64
    latent_dim: int = 30
    lr: float = 1e-3
    max_epochs: int = 200
    weight_decay: float = 0.0
    negative_slope: float = 0.2


class STAGATEMinimal:
    """
    Minimal, portable STAGATE-style baseline (graph-attention autoencoder).

    Notes
    -----
    - Uses only torch + sklearn, no scanpy/pyG dependencies.
    - Trains an attention autoencoder to reconstruct a PCA embedding of the
      log1p-HVG expression matrix.
    - Produces fixed-K labels by clustering the latent embedding with k-means.
    """

    def __init__(self, config: STAGATEMinimalConfig | None = None) -> None:
        self.config = config or STAGATEMinimalConfig()
        self.model: STAGATEAutoEncoderMinimal | None = None
        self.embed_pca: np.ndarray | None = None
        self.embed_latent: np.ndarray | None = None

    def fit(self, x: np.ndarray, spatial_adj, *, seed: int) -> None:
        x_pca = _pca_embed(x, self.config.num_pcs, seed=int(seed))

        src_np, dst_np = _edge_index_from_csr(spatial_adj)
        # Add self-loops for stability
        n = x_pca.shape[0]
        self_src = np.arange(n, dtype=np.int64)
        self_dst = np.arange(n, dtype=np.int64)
        src_np = np.concatenate([src_np, self_src], axis=0)
        dst_np = np.concatenate([dst_np, self_dst], axis=0)

        torch.manual_seed(int(seed))
        x_t = torch.as_tensor(x_pca, dtype=torch.float32)
        src_t = torch.as_tensor(src_np, dtype=torch.long)
        dst_t = torch.as_tensor(dst_np, dtype=torch.long)

        model = STAGATEAutoEncoderMinimal(
            in_dim=int(x_pca.shape[1]),
            hidden_dim=int(self.config.hidden_dim),
            latent_dim=int(self.config.latent_dim),
        )
        # Set leaky slope
        for layer in [model.enc1, model.enc2, model.dec1, model.dec2]:
            layer.negative_slope = float(self.config.negative_slope)

        opt = optim.Adam(model.parameters(), lr=float(self.config.lr), weight_decay=float(self.config.weight_decay))
        model.train()
        for _ in range(int(self.config.max_epochs)):
            opt.zero_grad(set_to_none=True)
            _, x_hat = model(x_t, src_t, dst_t)
            loss = torch.mean((x_hat - x_t) ** 2)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            z, _ = model(x_t, src_t, dst_t)

        self.model = model
        self.embed_pca = x_pca
        self.embed_latent = z.detach().cpu().numpy().astype(np.float32, copy=False)

    def predict_labels(self, *, k: int, seed: int) -> np.ndarray:
        if self.embed_latent is None:
            raise RuntimeError("STAGATEMinimal not trained")
        km = KMeans(n_clusters=int(k), n_init=20, random_state=int(seed), max_iter=300)
        labels = km.fit_predict(self.embed_latent)
        return labels.astype(int)
