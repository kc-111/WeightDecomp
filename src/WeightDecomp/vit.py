"""Vision Transformer (ViT) with centered-B DecomposedLinear.

The key insight (paper Section 4.5): mean-centering B across the neuron dimension
creates a nonlinear parameterization where dead neurons receive B-channel gradients:

    W_eff = W + (B - mean(B, dim=0)) @ C

For dead neuron j:
    ∂L/∂B[j,:] = -(1/m) Σ_{k alive} G[k,:] Cᵀ  ≠ 0

This breaks the bilinear symmetry where the B-channel is dead for dead neurons.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decomposed_linear import DecomposedLinear


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.norm(2, dim=-1, keepdim=True) / math.sqrt(x.size(-1))
        return self.scale * x / (rms + self.eps)


class CenteredDecomposedLinear(DecomposedLinear):
    """DecomposedLinear with mean-centered B for cross-neuron gradient coupling.

    W_eff = W + Σ (Bᵢ - mean(Bᵢ, dim=0)) @ Cᵢ

    The centering makes ∂L/∂B[j,:] nonzero for dead neurons, enabling
    B-channel revival that is absent in the standard bilinear case.
    """

    def effective_weight(self) -> torch.Tensor:
        W_eff = self.W
        for B, C in zip(self.Bs, self.Cs):
            B_centered = B - B.mean(dim=0, keepdim=True)
            W_eff = W_eff + B_centered @ C
        return W_eff

    def merge(self, rerandomize_B: bool = True) -> None:
        with torch.no_grad():
            for B, C in zip(self.Bs, self.Cs):
                B_centered = B - B.mean(dim=0, keepdim=True)
                self.W.add_(B_centered @ C)
                C.zero_()
            if rerandomize_B:
                for B in self.Bs:
                    nn.init.kaiming_uniform_(B, a=math.sqrt(5))


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, ranks: list[int] | None = None):
        super().__init__()
        self.fc1 = CenteredDecomposedLinear(d_model, d_ff, ranks=ranks)
        self.fc2 = CenteredDecomposedLinear(d_ff, d_model, ranks=ranks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 ranks: list[int] | None = None, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff, ranks=ranks)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(h)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class DecomposedViT(nn.Module):
    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
        num_classes: int = 10,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 128,
        ranks: list[int] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        n_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, ranks=ranks, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        p = self.patch_size

        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, -1, x.size(1) * p * p)
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x[:, 0])
        return self.head(x)

    def ffn_layers(self) -> list[CenteredDecomposedLinear]:
        return [block.ffn.fc1 for block in self.blocks] + \
            [block.ffn.fc2 for block in self.blocks]

    def merge_all(self, rerandomize_B: bool = True) -> None:
        for layer in self.ffn_layers():
            layer.merge(rerandomize_B=rerandomize_B)

    def split_all(self, ranks: list[int], rerandomize_B: bool = True) -> None:
        for layer in self.ffn_layers():
            layer.split(ranks, rerandomize_B=rerandomize_B)
