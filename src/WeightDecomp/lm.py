"""Decoder-only language model with weight decomposition.

Causal transformer with SharedFactorLinear for all projections.
Supports per-block FactorScope sharing (same as DecomposedViT).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import SharedFactorLinear, RMSNorm, FFN, TransformerBlock


class CausalAttention(nn.Module):
    """Multi-head self-attention with causal mask."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 1024,
                 dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0

        self.q_proj = SharedFactorLinear(d_model, d_model, bias=False)
        self.k_proj = SharedFactorLinear(d_model, d_model, bias=False)
        self.v_proj = SharedFactorLinear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # Causal mask (upper triangular = -inf)
        mask = torch.full((max_seq_len, max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Bsz, S, D = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(Bsz, S, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(Bsz, S, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(Bsz, S, self.n_heads, self.d_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = attn + self.causal_mask[:S, :S]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(Bsz, S, D)
        return self.out_proj(out)

    def projections(self):
        return [self.q_proj, self.k_proj, self.v_proj]


class CausalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 max_seq_len: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalAttention(d_model, n_heads, max_seq_len, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

    def all_projections(self):
        return self.attn.projections() + [self.ffn.fc1, self.ffn.fc2]


class DecomposedLM(nn.Module):
    """Decoder-only language model with weight decomposition.

    Args:
        vocab_size: Vocabulary size.
        d_model: Embedding / hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks.
        d_ff: FFN intermediate dimension.
        max_seq_len: Maximum sequence length.
        dropout: Dropout rate.
        tie_weights: Tie embedding and output head weights.
    """

    def __init__(self, vocab_size: int = 50257, d_model: int = 256,
                 n_heads: int = 4, n_layers: int = 6, d_ff: int = 512,
                 max_seq_len: int = 256, dropout: float = 0.1,
                 tie_weights: bool = True):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            CausalTransformerBlock(d_model, n_heads, d_ff, max_seq_len, dropout)
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.head.weight = self.tok_emb.weight

        self.scopes = nn.ModuleList()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, SharedFactorLinear)):
            if hasattr(module, 'W'):
                nn.init.normal_(module.W, std=0.02)
            elif hasattr(module, 'weight'):
                nn.init.normal_(module.weight, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits (B, S, vocab_size)."""
        B, S = x.shape
        pos = torch.arange(S, device=x.device)
        x = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)

    def ffn_layers(self):
        return [b.ffn.fc1 for b in self.blocks] + [b.ffn.fc2 for b in self.blocks]

    def split_all(self, shared_ranks: list[int],
                  local_ranks: list[int] | None = None,
                  sharing: str = "per_block",
                  b_transform: str = "identity") -> None:
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
        else:
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

    def merge_all(self, rerandomize_B: bool = True) -> None:
        for i in range(len(self.scopes)):
            self.scopes[i].merge(rerandomize_B=rerandomize_B)
        for block in self.blocks:
            for layer in block.all_projections():
                layer.merge_local(rerandomize_B=rerandomize_B)

    def reset_factor_optimizer_state(self, optimizer) -> None:
        for i in range(len(self.scopes)):
            self.scopes[i].reset_optimizer_state(optimizer)

    def compiled(self, mode: str = "reduce-overhead"):
        torch.set_float32_matmul_precision("high")
        return torch.compile(self, mode=mode, fullgraph=True)
