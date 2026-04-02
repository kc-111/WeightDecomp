import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecomposedLinear(nn.Module):
    """Linear layer with mixed-rank weight decomposition.

    Effective weight: W_eff = W + sum_i B_i @ C_i

    Each (B_i, C_i) factor pair has rank r_i:
        B_i in R^{out_features x r_i}
        C_i in R^{r_i x in_features}

    C_i is initialized to zero so W_eff = W at construction.
    The coupling matrix P = I + sum_i B_i B_i^T enables dead neuron
    revival through inter-neuron gradient coupling (paper Sec 6.1).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        ranks: list[int] | None = None,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.Bs = nn.ParameterList()
        self.Cs = nn.ParameterList()

        self._reset_parameters()

        if ranks:
            for r in ranks:
                self.add_factor(r)

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def num_factors(self) -> int:
        return len(self.Bs)

    @property
    def ranks(self) -> list[int]:
        return [B.shape[1] for B in self.Bs]

    def add_factor(self, rank: int, init_scale: float = 0.1) -> None:
        """Add a new (B, C) factor pair with given rank. C=0 so W_eff unchanged."""
        B = nn.Parameter(
            torch.randn(self.out_features, rank, device=self.W.device, dtype=self.W.dtype) * init_scale
        )
        C = nn.Parameter(
            torch.zeros(rank, self.in_features, device=self.W.device, dtype=self.W.dtype)
        )
        self.Bs.append(B)
        self.Cs.append(C)

    def remove_factors(self) -> None:
        """Remove all factor pairs (does NOT merge first)."""
        self.Bs = nn.ParameterList()
        self.Cs = nn.ParameterList()

    def effective_weight(self) -> torch.Tensor:
        """Compute W_eff = alpha * W + sum_i B_i @ C_i (differentiable)."""
        W_eff = self.alpha * self.W if self.alpha != 1.0 else self.W
        for B, C in zip(self.Bs, self.Cs):
            W_eff = W_eff + B @ C
        return W_eff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)

    def merge(self, rerandomize_B: bool = True) -> None:
        """Merge factors into W: W <- alpha*W + sum B_i C_i, then reset C_i = 0.

        Optionally re-randomize B to refresh coupling directions.
        The effective weight is unchanged by this operation.
        """
        with torch.no_grad():
            if self.alpha != 1.0:
                self.W.mul_(self.alpha)
                self.alpha = 1.0
            for B, C in zip(self.Bs, self.Cs):
                self.W.add_(B @ C)
                C.zero_()
            if rerandomize_B:
                for B in self.Bs:
                    B.normal_(0, 0.1)

    def split(self, ranks: list[int], rerandomize_B: bool = True) -> None:
        """Merge current factors into W, remove them, then add new factors."""
        if self.num_factors > 0:
            self.merge(rerandomize_B=False)
            self.remove_factors()
        for r in ranks:
            self.add_factor(r)

    def coupling_matrix(self) -> torch.Tensor:
        """Compute P = I + sum_i B_i B_i^T (detached, for diagnostics)."""
        with torch.no_grad():
            P = torch.eye(self.out_features, device=self.W.device, dtype=self.W.dtype)
            for B in self.Bs:
                P = P + B @ B.T
            return P

    def grow(self, new_out: int | None = None, new_in: int | None = None,
             init_scale: float = 0.1) -> None:
        """Grow the layer by adding new neurons. Function-preserving:
        new output rows have negative random weights (pre-ReLU → dead start),
        new input columns are zero (no signal from new upstream neurons yet).

        New neurons are initialized at the same alpha scale as existing factors.

        Args:
            new_out: new total out_features (must be >= current). None = no change.
            new_in: new total in_features (must be >= current). None = no change.
            init_scale: scale for new output neuron weights (negative, random).
        """
        new_out = new_out or self.out_features
        new_in = new_in or self.in_features
        assert new_out >= self.out_features and new_in >= self.in_features

        d_out = new_out - self.out_features
        d_in = new_in - self.in_features

        with torch.no_grad():
            dev, dtype = self.W.device, self.W.dtype

            if d_out > 0 or d_in > 0:
                # Grow W: new rows at alpha-scaled init, new cols = zero
                W_new = torch.zeros(new_out, new_in, device=dev, dtype=dtype)
                W_new[:self.out_features, :self.in_features] = self.W.data
                # New rows = zero. Factors break symmetry.
                self.W = nn.Parameter(W_new)

                # Grow bias
                if self.bias is not None and d_out > 0:
                    b_new = torch.zeros(new_out, device=dev, dtype=dtype)
                    b_new[:self.out_features] = self.bias.data
                    # New bias = zero (function preserving)
                    self.bias = nn.Parameter(b_new)

                # Grow factors
                for i in range(len(self.Bs)):
                    B_old = self.Bs[i].data
                    C_old = self.Cs[i].data
                    r = B_old.shape[1]

                    if d_out > 0:
                        B_ext = torch.randn(d_out, r, device=dev, dtype=dtype) * init_scale
                        B_new = torch.cat([B_old, B_ext], dim=0)
                        self.Bs[i] = nn.Parameter(B_new)

                    if d_in > 0:
                        C_ext = torch.zeros(r, d_in, device=dev, dtype=dtype)
                        C_new = torch.cat([C_old, C_ext], dim=1)
                        self.Cs[i] = nn.Parameter(C_new)

            self.out_features = new_out
            self.in_features = new_in

    def extra_repr(self) -> str:
        ranks_str = f", ranks={self.ranks}" if self.num_factors > 0 else ""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}{ranks_str}"
        )
