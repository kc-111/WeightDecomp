"""FactorScope: owns shared B and C parameters, attaches them to layers.

Supports configurable B transforms:
  - "identity": B used as-is (standard bilinear ABC)
  - "centered": B - mean(B, dim=0) (cross-row coupling for dead neurons)
  - "cayley":   Cayley(S) where S = (B - B^T)/2 (orthogonal, dense coupling)
"""

import math
from typing import Literal

import torch
import torch.nn as nn

BTransform = Literal["identity", "centered", "cayley"]


def _apply_b_transform(B: torch.Tensor, transform: str) -> torch.Tensor:
    """Apply the configured transform to a shared B parameter."""
    if transform == "identity":
        return B
    elif transform == "centered":
        return B - B.mean(dim=0, keepdim=True)
    elif transform == "cayley":
        S = (B - B.T) / 2  # skew-symmetric
        r = S.shape[0]
        I = torch.eye(r, device=S.device, dtype=S.dtype)
        return torch.linalg.solve(I - S, I + S)
    else:
        raise ValueError(f"Unknown b_transform: {transform}")


class FactorScope(nn.Module):
    """Owns shared B (r×r) and C (r×d_in) parameters for a group of layers.

    B is shared across all registered layers.
    C is shared across layers with the same input dimension.
    Each layer gets its own A (out×r) when registered.

    Args:
        ranks: List of ranks for each factor.
        b_transform: Transform applied to B in effective_weight.
            "identity" — B used directly (default)
            "centered" — B - mean(B, dim=0), opens B-channel for dead neurons
            "cayley"   — Cayley transform, orthogonal with dense coupling
        device, dtype: For parameter creation.
    """

    def __init__(self, ranks: list[int], b_transform: BTransform = "identity",
                 device=None, dtype=None):
        super().__init__()
        self.ranks = list(ranks)
        self.b_transform = b_transform
        self._layers: list[nn.Module] = []

        # Shared B: one per rank (r × r)
        self.shared_Bs = nn.ParameterList()
        for r in ranks:
            B = nn.Parameter(torch.empty(r, r, device=device, dtype=dtype))
            nn.init.orthogonal_(B)
            self.shared_Bs.append(B)

        # Shared C: keyed by input_dim
        self._shared_Cs = nn.ModuleDict()

    def _get_or_create_Cs(self, in_features: int, device=None, dtype=None):
        key = str(in_features)
        if key not in self._shared_Cs:
            Cs = nn.ParameterList()
            for r in self.ranks:
                C = nn.Parameter(torch.zeros(r, in_features, device=device, dtype=dtype))
                Cs.append(C)
            self._shared_Cs[key] = Cs
        return self._shared_Cs[key]

    def get_transformed_Bs(self) -> list[torch.Tensor]:
        """Return B matrices with the configured transform applied."""
        return [_apply_b_transform(self.shared_Bs[i], self.b_transform)
                for i in range(len(self.shared_Bs))]

    def register_layer(self, layer) -> None:
        """Register a SharedFactorLinear layer with this scope."""
        from .vit import SharedFactorLinear
        assert isinstance(layer, SharedFactorLinear)

        dev, dtype = layer.W.device, layer.W.dtype
        Cs = self._get_or_create_Cs(layer.in_features, dev, dtype)

        layer.remove_shared_factors()
        for r in self.ranks:
            layer.add_shared_factor(r)

        # Attach scope reference so layer can call get_transformed_Bs()
        layer._factor_scope = self
        layer._shared_Cs = [Cs[i] for i in range(len(Cs))]
        self._layers.append(layer)

    def merge(self, rerandomize_B: bool = True) -> None:
        """Merge shared ABC into W for all registered layers."""
        # Compute transformed Bs ONCE for merge
        transformed_Bs = self.get_transformed_Bs()
        for layer in self._layers:
            layer.merge_shared(transformed_Bs)

        with torch.no_grad():
            for key in self._shared_Cs:
                Cs = self._shared_Cs[key]
                for i in range(len(Cs)):
                    Cs[i].zero_()
            if rerandomize_B:
                for i in range(len(self.shared_Bs)):
                    nn.init.orthogonal_(self.shared_Bs[i])

    def reset_optimizer_state(self, optimizer: torch.optim.Optimizer) -> None:
        """Remove stale optimizer state for all factor params."""
        params_to_reset = []
        for i in range(len(self.shared_Bs)):
            params_to_reset.append(self.shared_Bs[i])
        for key in self._shared_Cs:
            Cs = self._shared_Cs[key]
            for i in range(len(Cs)):
                params_to_reset.append(Cs[i])
        for layer in self._layers:
            for i in range(len(layer.As)):
                params_to_reset.append(layer.As[i])
            for i in range(len(layer.Bs_local)):
                params_to_reset.append(layer.Bs_local[i])
            for i in range(len(layer.Cs_local)):
                params_to_reset.append(layer.Cs_local[i])
        for param in params_to_reset:
            if param in optimizer.state:
                del optimizer.state[param]
