"""Hurwitz-decomposed linear layer.

W_eff = alpha * W + A @ H @ D

where H = Σ_k (B_k B_k^T + C_k - C_k^T) is a Hurwitz matrix:
  - B_k B_k^T: symmetric PSD (rank r per component)
  - C_k - C_k^T: skew-symmetric (parameterized via strict lower triangular)

D is zero-initialized so W_eff = alpha * W at construction (function-preserving).
The Hurwitz structure gives a much richer preconditioner than plain BC,
providing dense inter-neuron coupling for dead neuron revival.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flat_to_skew_batch(flat_batch: torch.Tensor, N: int) -> torch.Tensor:
    """Convert (K, N*(N-1)/2) flat vectors to (K, N, N) skew-symmetric matrices."""
    K = flat_batch.shape[0]
    idx = torch.tril_indices(N, N, offset=-1, device=flat_batch.device)
    L = torch.zeros(K, N, N, device=flat_batch.device, dtype=flat_batch.dtype)
    L[:, idx[0], idx[1]] = flat_batch
    return L - L.transpose(1, 2)


class HurwitzLinear(nn.Module):
    """Linear layer: W_eff = alpha * W + A @ H @ D

    H = Σ_k (B_k B_k^T + Skew_k) is Hurwitz (all eigenvalues have Re > 0).

    Args:
        in_features, out_features: layer dimensions.
        r: rank of each B_k component.
        K: number of Hurwitz components summed.
        alpha: scaling on the base weight W.
        bias: whether to include bias.
    """

    def __init__(self, in_features: int, out_features: int,
                 r: int = 4, K: int = 4, alpha: float = 1.0,
                 bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.K = K
        self.alpha = alpha
        N = out_features

        # Base weight
        self.W = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Hurwitz components
        comp_scale = 0.1 / math.sqrt(K)
        self.Bs = nn.Parameter(torch.randn(K, N, r) * comp_scale)
        self.M_flat = nn.Parameter(torch.randn(K, N * (N - 1) // 2) * comp_scale)

        # Sandwich matrices
        self.A = nn.Parameter(torch.randn(N, N) * 0.1)
        self.D = nn.Parameter(torch.zeros(N, in_features))  # zero init → function preserving

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def _hurwitz_matrix(self) -> torch.Tensor:
        """Compute H = Σ_k (B_k B_k^T + Skew_k) — fully batched, no loops."""
        # Bs: (K, N, r) → bmm → (K, N, N) → sum → (N, N)
        sym = torch.bmm(self.Bs, self.Bs.transpose(1, 2)).sum(0)
        skew = _flat_to_skew_batch(self.M_flat, self.out_features).sum(0)
        return sym + skew

    def effective_weight(self) -> torch.Tensor:
        H = self._hurwitz_matrix()
        W_eff = self.alpha * self.W + self.A @ H @ self.D
        self._cached_w_eff = W_eff
        return W_eff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)

    def merge(self, rerandomize: bool = True) -> None:
        """Absorb A @ H @ D into W, reset D to zero."""
        with torch.no_grad():
            H = self._hurwitz_matrix()
            self.W.mul_(self.alpha)
            self.W.add_(self.A @ H @ self.D)
            self.D.zero_()
            if rerandomize:
                comp_scale = 0.1 / math.sqrt(self.K)
                self.Bs.normal_(0, comp_scale)
                self.M_flat.normal_(0, comp_scale)
                self.A.normal_(0, 0.1)

    def grow(self, new_out: int | None = None, new_in: int | None = None,
             init_scale: float = 0.1) -> None:
        """Grow the layer. New output neurons start with negative weights."""
        new_out = new_out or self.out_features
        new_in = new_in or self.in_features
        assert new_out >= self.out_features and new_in >= self.in_features

        d_out = new_out - self.out_features
        d_in = new_in - self.in_features

        with torch.no_grad():
            dev, dtype = self.W.device, self.W.dtype
            N_old = self.out_features
            comp_scale = init_scale / math.sqrt(self.K)

            if d_out > 0 or d_in > 0:
                # Grow W: new rows = zero. Factors break symmetry.
                W_new = torch.zeros(new_out, new_in, device=dev, dtype=dtype)
                W_new[:N_old, :self.in_features] = self.W.data
                self.W = nn.Parameter(W_new)

                # Grow bias
                if self.bias is not None and d_out > 0:
                    b_new = torch.zeros(new_out, device=dev, dtype=dtype)
                    b_new[:N_old] = self.bias.data
                    # New bias = zero (function preserving)
                    self.bias = nn.Parameter(b_new)

                # Grow Bs: (K, N, r) → (K, N+d_out, r)
                if d_out > 0:
                    Bs_ext = torch.randn(self.K, d_out, self.r, device=dev, dtype=dtype) * comp_scale
                    self.Bs = nn.Parameter(torch.cat([self.Bs.data, Bs_ext], dim=1))

                # Grow M_flat: N*(N-1)/2 → N_new*(N_new-1)/2
                if d_out > 0:
                    N_new = new_out
                    new_skew_size = N_new * (N_new - 1) // 2
                    M_new = torch.randn(self.K, new_skew_size, device=dev, dtype=dtype) * comp_scale
                    # Copy old entries (they correspond to the first N_old neurons)
                    old_skew_size = N_old * (N_old - 1) // 2
                    M_new[:, :old_skew_size] = self.M_flat.data
                    self.M_flat = nn.Parameter(M_new)

                # Grow A: (N, N) → (N_new, N_new)
                if d_out > 0:
                    A_new = torch.randn(new_out, new_out, device=dev, dtype=dtype) * init_scale
                    A_new[:N_old, :N_old] = self.A.data
                    self.A = nn.Parameter(A_new)

                # Grow D: (N, in) → (N_new, in_new)
                D_new = torch.zeros(new_out, new_in, device=dev, dtype=dtype)
                D_new[:N_old, :self.in_features] = self.D.data
                self.D = nn.Parameter(D_new)

            self.out_features = new_out
            self.in_features = new_in

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"r={self.r}, K={self.K}, alpha={self.alpha}, "
                f"bias={self.bias is not None}")


class HurwitzMLP(nn.Module):
    """MLP using HurwitzLinear layers."""

    def __init__(self, layer_sizes: list[int], r: int = 4, K: int = 4,
                 alpha: float = 1.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                HurwitzLinear(layer_sizes[i], layer_sizes[i + 1],
                              r=r, K=K, alpha=alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

    def decomposed_layers(self):
        return list(self.layers)

    def merge_all(self, rerandomize: bool = True) -> None:
        for layer in self.layers:
            layer.merge(rerandomize=rerandomize)

    def grow_width(self, new_hidden: int, init_scale: float = 0.1) -> None:
        """Grow all hidden layers to new_hidden width."""
        for i, layer in enumerate(self.layers):
            is_first = i == 0
            is_last = i == len(self.layers) - 1
            new_out = layer.out_features if is_last else new_hidden
            new_in = layer.in_features if is_first else new_hidden
            if new_out != layer.out_features or new_in != layer.in_features:
                layer.grow(new_out=new_out, new_in=new_in, init_scale=init_scale)


# ========== Hurwitz ViT ==========

class _RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rms = x.norm(2, dim=-1, keepdim=True) / math.sqrt(x.size(-1))
        return self.scale * x / (rms + self.eps)


class HurwitzAttention(nn.Module):
    def __init__(self, d_model, n_heads, r=4, K=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0
        self.q_proj = HurwitzLinear(d_model, d_model, r=r, K=K, alpha=alpha, bias=False)
        self.k_proj = HurwitzLinear(d_model, d_model, r=r, K=K, alpha=alpha, bias=False)
        self.v_proj = HurwitzLinear(d_model, d_model, r=r, K=K, alpha=alpha, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        Bsz, S, D = x.shape
        q = self.q_proj(x).view(Bsz, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(Bsz, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(Bsz, S, self.n_heads, self.d_head).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = self.attn_dropout(F.softmax(attn, dim=-1))
        return self.out_proj((attn @ v).transpose(1, 2).contiguous().view(Bsz, S, D))

    def projections(self):
        return [self.q_proj, self.k_proj, self.v_proj]


class HurwitzFFN(nn.Module):
    def __init__(self, d_model, d_ff, r=4, K=4, alpha=1.0):
        super().__init__()
        self.fc1 = HurwitzLinear(d_model, d_ff, r=r, K=K, alpha=alpha)
        self.fc2 = HurwitzLinear(d_ff, d_model, r=r, K=K, alpha=alpha)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class HurwitzTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, r=4, K=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.norm1 = _RMSNorm(d_model)
        self.attn = HurwitzAttention(d_model, n_heads, r, K, alpha, dropout)
        self.norm2 = _RMSNorm(d_model)
        self.ffn = HurwitzFFN(d_model, d_ff, r, K, alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

    def all_projections(self):
        return self.attn.projections() + [self.ffn.fc1, self.ffn.fc2]


class HurwitzViT(nn.Module):
    """ViT with Hurwitz decomposition on all projections."""

    def __init__(self, img_size=28, patch_size=7, in_channels=1, num_classes=10,
                 d_model=64, n_heads=4, n_layers=4, d_ff=128,
                 r=4, K=4, alpha=1.0, dropout=0.0):
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
            HurwitzTransformerBlock(d_model, n_heads, d_ff, r, K, alpha, dropout)
            for _ in range(n_layers)
        ])
        self.norm = _RMSNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B = x.size(0); p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, -1, x.size(1) * p * p)
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.dropout_layer(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 0]))

    def ffn_layers(self):
        return [b.ffn.fc1 for b in self.blocks] + [b.ffn.fc2 for b in self.blocks]

    def decomposed_layers(self):
        layers = []
        for b in self.blocks:
            layers.extend(b.all_projections())
        return layers

    def merge_all(self, rerandomize: bool = True):
        for layer in self.decomposed_layers():
            layer.merge(rerandomize=rerandomize)

    def grow_width(self, new_d_model: int | None = None,
                   new_d_ff: int | None = None) -> None:
        """Grow d_model and/or d_ff. Handles all layers, embeddings, norms."""
        new_d = new_d_model or self.d_model
        new_ff = new_d_ff or self.d_ff
        d_d = new_d - self.d_model
        d_ff = new_ff - self.d_ff
        dev = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        if d_d > 0:
            # Patch embedding: new rows = zero
            old_w = self.patch_embed.weight.data
            old_b = self.patch_embed.bias.data
            self.patch_embed = nn.Linear(old_w.shape[1], new_d).to(dev)
            nn.init.zeros_(self.patch_embed.weight)
            nn.init.zeros_(self.patch_embed.bias)
            self.patch_embed.weight.data[:self.d_model] = old_w
            self.patch_embed.bias.data[:self.d_model] = old_b

            # CLS token: (1, 1, d_model) → (1, 1, new_d)
            cls_new = torch.zeros(1, 1, new_d, device=dev, dtype=dtype)
            cls_new[:, :, :self.d_model] = self.cls_token.data
            self.cls_token = nn.Parameter(cls_new)

            # Pos embed: (1, n_patches+1, d_model) → (1, n_patches+1, new_d)
            pos_new = torch.zeros(1, self.pos_embed.shape[1], new_d, device=dev, dtype=dtype)
            pos_new[:, :, :self.d_model] = self.pos_embed.data
            self.pos_embed = nn.Parameter(pos_new)

            # Head: new input cols = zero
            old_w = self.head.weight.data
            old_b = self.head.bias.data
            self.head = nn.Linear(new_d, old_w.shape[0]).to(dev)
            nn.init.zeros_(self.head.weight)
            self.head.weight.data[:, :self.d_model] = old_w
            self.head.bias.data = old_b

            # Final norm
            rms_comp = math.sqrt(self.d_model) / math.sqrt(new_d)
            scale_new = torch.ones(new_d, device=dev, dtype=dtype)
            scale_new[:self.d_model] = self.norm.scale.data * rms_comp
            self.norm.scale = nn.Parameter(scale_new)

        # Grow each block
        for block in self.blocks:
            if d_d > 0:
                # Attention QKV: grow both in and out
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
                block.attn.n_heads = new_d // block.attn.d_head

                # Norms
                for norm in [block.norm1, block.norm2]:
                    s_new = torch.ones(new_d, device=dev, dtype=dtype)
                    s_new[:self.d_model] = norm.scale.data * rms_comp
                    norm.scale = nn.Parameter(s_new)

            # FFN fc1: (d_model→d_ff) → (new_d→new_ff)
            block.ffn.fc1.grow(
                new_out=new_ff if d_ff > 0 else None,
                new_in=new_d if d_d > 0 else None)
            # FFN fc2: (d_ff→d_model) → (new_ff→new_d)
            block.ffn.fc2.grow(
                new_out=new_d if d_d > 0 else None,
                new_in=new_ff if d_ff > 0 else None)

        self.d_model = new_d
        self.d_ff = new_ff
        print(f"  Grew d_model={self.d_model} d_ff={self.d_ff}, "
              f"params: {sum(p.numel() for p in self.parameters()):,}")

    def compiled(self, mode="reduce-overhead"):
        torch.set_float32_matmul_precision("high")
        return torch.compile(self, mode=mode, fullgraph=True)
