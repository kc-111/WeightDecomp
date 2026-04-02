"""Vision Transformer (ViT) with per-block shared factor decomposition.

Each TransformerBlock has a FactorScope that owns shared B(r×r) and C(r×d_in).
All projections (Q, K, V, fc1, fc2) within a block share B and C (by input dim).
Layers hold references — no shared args passed through forward().

Architecture per layer: W_eff = W + Σ A_i @ B_shared_i @ C_shared_i + Σ B_local_j @ C_local_j
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.norm(2, dim=-1, keepdim=True) / math.sqrt(x.size(-1))
        return self.scale * x / (rms + self.eps)


class SharedFactorLinear(nn.Module):
    """Linear layer with shared ABC + local BC decomposition.

    Shared factors (A, B_ref, C_ref) are attached by a FactorScope.
    Local factors (B, C) are owned directly.
    No shared args in forward() — uses stored references.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 alpha: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.W = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Per-layer A for shared factors (attached by FactorScope)
        self.As = nn.ParameterList()
        # Reference to owning FactorScope (set by FactorScope.register_layer)
        self._factor_scope = None
        self._shared_Cs: list[torch.Tensor] = []
        # Local factors for optimization
        self.Bs_local = nn.ParameterList()
        self.Cs_local = nn.ParameterList()

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def add_shared_factor(self, rank: int) -> None:
        """Add a per-layer A factor (paired with scope's shared B/C)."""
        dev, dtype = self.W.device, self.W.dtype
        A = nn.Parameter(torch.randn(self.out_features, rank, device=dev, dtype=dtype) * 0.1)
        self.As.append(A)

    def add_local_factor(self, rank: int) -> None:
        """Add a standard (B, C) factor pair for optimization."""
        dev, dtype = self.W.device, self.W.dtype
        B = nn.Parameter(torch.randn(self.out_features, rank, device=dev, dtype=dtype) * 0.1)
        C = nn.Parameter(torch.zeros(rank, self.in_features, device=dev, dtype=dtype))
        self.Bs_local.append(B)
        self.Cs_local.append(C)

    def remove_shared_factors(self) -> None:
        self.As = nn.ParameterList()
        self._factor_scope = None
        self._shared_Cs = []

    def remove_local_factors(self) -> None:
        self.Bs_local = nn.ParameterList()
        self.Cs_local = nn.ParameterList()

    def remove_factors(self) -> None:
        self.remove_shared_factors()
        self.remove_local_factors()

    def effective_weight(self) -> torch.Tensor:
        W_eff = self.alpha * self.W if self.alpha != 1.0 else self.W
        if self._factor_scope is not None:
            transformed_Bs = self._factor_scope.get_transformed_Bs()
            for i in range(len(self.As)):
                W_eff = W_eff + self.As[i] @ transformed_Bs[i] @ self._shared_Cs[i]
        for i in range(len(self.Bs_local)):
            W_eff = W_eff + self.Bs_local[i] @ self.Cs_local[i]
        return W_eff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)

    def merge_shared(self, transformed_Bs: list[torch.Tensor]) -> None:
        """Merge shared A@B@C into W and re-init A. Called by FactorScope."""
        with torch.no_grad():
            # W_eff was alpha*W + Σ A@B@C. Make W_new = W_eff / alpha_new.
            # Since alpha_new will be 1.0 after full merge, W_new = alpha*W + factors.
            if self.alpha != 1.0:
                self.W.mul_(self.alpha)
                self.alpha = 1.0
            for i in range(len(self.As)):
                self.W.add_(self.As[i] @ transformed_Bs[i] @ self._shared_Cs[i])
                self.As[i].normal_(0, 0.1)

    def merge_local(self, rerandomize_B: bool = True) -> None:
        """Merge local B@C into W."""
        with torch.no_grad():
            if self.alpha != 1.0:
                self.W.mul_(self.alpha)
                self.alpha = 1.0
            for i in range(len(self.Bs_local)):
                self.W.add_(self.Bs_local[i] @ self.Cs_local[i])
                self.Cs_local[i].zero_()
            if rerandomize_B:
                for i in range(len(self.Bs_local)):
                    self.Bs_local[i].normal_(0, 0.1)


    def grow(self, new_out: int | None = None, new_in: int | None = None) -> None:
        """Grow layer dimensions. New rows = alpha-scaled Kaiming, new cols = zero."""
        new_out = new_out or self.out_features
        new_in = new_in or self.in_features
        d_out = new_out - self.out_features
        d_in = new_in - self.in_features
        if d_out <= 0 and d_in <= 0:
            return

        with torch.no_grad():
            dev, dtype = self.W.device, self.W.dtype

            W_new = torch.zeros(new_out, new_in, device=dev, dtype=dtype)
            W_new[:self.out_features, :self.in_features] = self.W.data
            # New rows = zero. Factors break symmetry.
            self.W = nn.Parameter(W_new)

            if self.bias is not None and d_out > 0:
                b_new = torch.zeros(new_out, device=dev, dtype=dtype)
                b_new[:self.out_features] = self.bias.data
                # New bias = zero (function preserving)
                self.bias = nn.Parameter(b_new)

            for i in range(len(self.As)):
                A = self.As[i].data
                r = A.shape[1]
                if d_out > 0:
                    A_new = torch.cat([A, torch.randn(d_out, r, device=dev, dtype=dtype) * 0.1], dim=0)
                    self.As[i] = nn.Parameter(A_new)

            for i in range(len(self.Bs_local)):
                B = self.Bs_local[i].data
                C = self.Cs_local[i].data
                r = B.shape[1]
                if d_out > 0:
                    self.Bs_local[i] = nn.Parameter(
                        torch.cat([B, torch.randn(d_out, r, device=dev, dtype=dtype) * 0.1], dim=0))
                if d_in > 0:
                    self.Cs_local[i] = nn.Parameter(
                        torch.cat([C, torch.zeros(r, d_in, device=dev, dtype=dtype)], dim=1))

        self.out_features = new_out
        self.in_features = new_in


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 alpha: float = 1.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0

        self.q_proj = SharedFactorLinear(d_model, d_model, bias=False, alpha=alpha)
        self.k_proj = SharedFactorLinear(d_model, d_model, bias=False, alpha=alpha)
        self.v_proj = SharedFactorLinear(d_model, d_model, bias=False, alpha=alpha)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Bsz, S, D = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(Bsz, S, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(Bsz, S, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(Bsz, S, self.n_heads, self.d_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(Bsz, S, D)
        return self.out_proj(out)

    def projections(self) -> list[SharedFactorLinear]:
        return [self.q_proj, self.k_proj, self.v_proj]


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, alpha: float = 1.0):
        super().__init__()
        self.fc1 = SharedFactorLinear(d_model, d_ff, alpha=alpha)
        self.fc2 = SharedFactorLinear(d_ff, d_model, alpha=alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.0, skip: bool = False, alpha: float = 1.0):
        super().__init__()
        self.skip = skip
        self.norm1 = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout=dropout, alpha=alpha)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff, alpha=alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn(self.norm1(x))
        x = x + self.dropout(h) if self.skip else self.dropout(h)
        h = self.ffn(self.norm2(x))
        x = x + self.dropout(h) if self.skip else self.dropout(h)
        return x

    def all_projections(self) -> list[SharedFactorLinear]:
        """All SharedFactorLinear layers in this block."""
        return self.attn.projections() + [self.ffn.fc1, self.ffn.fc2]


class DecomposedViT(nn.Module):
    """ViT with per-block FactorScope for shared factor decomposition."""

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
        skip: bool = False,
        alpha: float = 1.0,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.d_model = d_model
        self.d_ff = d_ff
        n_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout_layer = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout=dropout, skip=skip,
                             alpha=alpha)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        # FactorScopes — created by split_all()
        self.scopes = nn.ModuleList()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, -1, x.size(1) * p * p)
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout_layer(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x[:, 0])
        return self.head(x)

    def fc1_layers(self) -> list[SharedFactorLinear]:
        return [block.ffn.fc1 for block in self.blocks]

    def fc2_layers(self) -> list[SharedFactorLinear]:
        return [block.ffn.fc2 for block in self.blocks]

    def ffn_layers(self) -> list[SharedFactorLinear]:
        return self.fc1_layers() + self.fc2_layers()

    def split_all(self, shared_ranks: list[int],
                  local_ranks: list[int] | None = None,
                  sharing: str = "per_block",
                  b_transform: str = "identity") -> None:
        """Create FactorScopes and register projections.

        Args:
            shared_ranks: Ranks for shared A@B@C factors.
            local_ranks: Ranks for per-layer B@C factors (optimization).
            sharing: "per_block" (one scope per block) or "global" (one scope).
            b_transform: "identity", "centered", or "cayley".
        """
        from .factor_scope import FactorScope

        dev = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        self.scopes = nn.ModuleList()

        if sharing == "global":
            scope = FactorScope(shared_ranks, b_transform=b_transform,
                                device=dev, dtype=dtype)
            for block in self.blocks:
                for layer in block.all_projections():
                    layer.remove_factors()
                    scope.register_layer(layer)
                    if local_ranks:
                        for r in local_ranks:
                            layer.add_local_factor(r)
            self.scopes.append(scope)

        elif sharing == "per_block":
            for block in self.blocks:
                scope = FactorScope(shared_ranks, b_transform=b_transform,
                                    device=dev, dtype=dtype)
                for layer in block.all_projections():
                    layer.remove_factors()
                    scope.register_layer(layer)
                    if local_ranks:
                        for r in local_ranks:
                            layer.add_local_factor(r)
                self.scopes.append(scope)

        else:
            raise ValueError(f"Unknown sharing mode: {sharing}")

    def merge_all(self, rerandomize_B: bool = True) -> None:
        """Merge all factors across all blocks."""
        for i in range(len(self.scopes)):
            self.scopes[i].merge(rerandomize_B=rerandomize_B)
        # Also merge local factors
        for block in self.blocks:
            for layer in block.all_projections():
                layer.merge_local(rerandomize_B=rerandomize_B)

    def reset_factor_optimizer_state(self, optimizer: torch.optim.Optimizer) -> None:
        """Reset optimizer state for all factor parameters."""
        for i in range(len(self.scopes)):
            self.scopes[i].reset_optimizer_state(optimizer)

    def dead_neuron_summary(self, dead_counts: dict) -> str:
        """Compact dead neuron summary for deep models."""
        fc1_dead = []
        fc2_dead = []
        n_blocks = len(self.blocks)
        for i, (d, t) in dead_counts.items():
            if i < n_blocks:
                fc1_dead.append(d)
            else:
                fc2_dead.append(d)

        fc1_total = sum(fc1_dead)
        fc1_cap = len(fc1_dead) * (self.d_ff if fc1_dead else 0)
        fc2_total = sum(fc2_dead)
        fc2_cap = len(fc2_dead) * (self.d_model if fc2_dead else 0)

        parts = [f"fc1: {fc1_total}/{fc1_cap}"]
        if fc1_dead:
            parts.append(f"({min(fc1_dead)}-{max(fc1_dead)}/blk)")
        parts.append(f"| fc2: {fc2_total}/{fc2_cap}")
        if fc2_dead:
            parts.append(f"({min(fc2_dead)}-{max(fc2_dead)}/blk)")
        return " ".join(parts)

    def effective_weight_l2(self) -> torch.Tensor:
        """L2 penalty on effective weights ||W_eff||² + ||bias||² for all layers.

        Penalizes the COMBINED weight W + A@B@C + B_local@C_local, not
        individual factors. Differentiable — gradients flow to W, A, B, C.
        """
        l2 = torch.tensor(0.0, device=next(self.parameters()).device)
        for block in self.blocks:
            for layer in block.all_projections():
                l2 = l2 + layer.effective_weight().pow(2).sum()
                if layer.bias is not None:
                    l2 = l2 + layer.bias.pow(2).sum()
        return l2

    def grow_width(self, new_d_model: int | None = None,
                   new_d_ff: int | None = None) -> None:
        """Grow d_model and/or d_ff across all layers.

        d_model growth must be a multiple of n_heads (adds new heads,
        keeps existing head dimensions intact).
        """
        new_d = new_d_model or self.d_model
        new_ff = new_d_ff or self.d_ff
        d_d = new_d - self.d_model
        d_ff = new_ff - self.d_ff
        n_heads = self.blocks[0].attn.n_heads
        if d_d > 0:
            d_head = self.d_model // n_heads
            assert new_d % d_head == 0, \
                f"new_d_model ({new_d}) must be divisible by d_head ({d_head})"
        dev = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        if d_d > 0:
            # Patch embed: new rows = zero
            old_w, old_b = self.patch_embed.weight.data, self.patch_embed.bias.data
            self.patch_embed = nn.Linear(old_w.shape[1], new_d).to(dev)
            nn.init.zeros_(self.patch_embed.weight)
            nn.init.zeros_(self.patch_embed.bias)
            self.patch_embed.weight.data[:self.d_model] = old_w
            self.patch_embed.bias.data[:self.d_model] = old_b

            # CLS + pos embed
            cls_new = torch.zeros(1, 1, new_d, device=dev, dtype=dtype)
            cls_new[:, :, :self.d_model] = self.cls_token.data
            self.cls_token = nn.Parameter(cls_new)

            pos_new = torch.zeros(1, self.pos_embed.shape[1], new_d, device=dev, dtype=dtype)
            pos_new[:, :, :self.d_model] = self.pos_embed.data
            self.pos_embed = nn.Parameter(pos_new)

            # Head: new INPUT cols = zero (no signal from new dims yet)
            old_w, old_b = self.head.weight.data, self.head.bias.data
            self.head = nn.Linear(new_d, old_w.shape[0]).to(dev)
            nn.init.zeros_(self.head.weight)
            self.head.weight.data[:, :self.d_model] = old_w
            self.head.bias.data = old_b

            # Final norm — compensate for RMS denominator change
            rms_comp = math.sqrt(self.d_model) / math.sqrt(new_d)
            s_new = torch.ones(new_d, device=dev, dtype=dtype)
            s_new[:self.d_model] = self.norm.scale.data * rms_comp
            self.norm.scale = nn.Parameter(s_new)

        for block in self.blocks:
            if d_d > 0:
                for proj in block.attn.projections():
                    proj.grow(new_out=new_d, new_in=new_d)
                # out_proj: all new entries = zero
                old_w = block.attn.out_proj.weight.data
                old_b = block.attn.out_proj.bias.data
                block.attn.out_proj = nn.Linear(new_d, new_d).to(dev)
                nn.init.zeros_(block.attn.out_proj.weight)
                nn.init.zeros_(block.attn.out_proj.bias)
                block.attn.out_proj.weight.data[:self.d_model, :self.d_model] = old_w
                block.attn.out_proj.bias.data[:self.d_model] = old_b
                block.attn.n_heads = new_d // block.attn.d_head  # add new heads, keep d_head

                for norm in [block.norm1, block.norm2]:
                    s_new = torch.ones(new_d, device=dev, dtype=dtype)
                    s_new[:self.d_model] = norm.scale.data * rms_comp
                    norm.scale = nn.Parameter(s_new)

            block.ffn.fc1.grow(
                new_out=new_ff if d_ff > 0 else None,
                new_in=new_d if d_d > 0 else None)
            block.ffn.fc2.grow(
                new_out=new_d if d_d > 0 else None,
                new_in=new_ff if d_ff > 0 else None)

        # Update shared C dimensions in FactorScopes
        if d_d > 0 or d_ff > 0:
            for scope in self.scopes:
                for key in list(scope._shared_Cs.keys()):
                    old_dim = int(key)
                    if old_dim == self.d_model and d_d > 0:
                        new_dim = new_d
                    elif old_dim == self.d_ff and d_ff > 0:
                        new_dim = new_ff
                    else:
                        continue
                    Cs = scope._shared_Cs[key]
                    new_Cs = nn.ParameterList()
                    for i in range(len(Cs)):
                        C_old = Cs[i].data
                        C_new = torch.zeros(C_old.shape[0], new_dim,
                                            device=dev, dtype=dtype)
                        C_new[:, :old_dim] = C_old
                        new_Cs.append(nn.Parameter(C_new))
                    del scope._shared_Cs[key]
                    scope._shared_Cs[str(new_dim)] = new_Cs
                    # Update layer references
                    for layer in scope._layers:
                        if layer.in_features == new_dim:
                            layer._shared_Cs = [new_Cs[j] for j in range(len(new_Cs))]

        self.d_model = new_d
        self.d_ff = new_ff
        print(f"  Grew d_model={new_d} d_ff={new_ff}, "
              f"params: {sum(p.numel() for p in self.parameters()):,}")

    def compiled(self, mode: str = "reduce-overhead") -> "DecomposedViT":
        """Return torch.compiled version. Call AFTER split_all()."""
        torch.set_float32_matmul_precision("high")
        return torch.compile(self, mode=mode, fullgraph=True)
