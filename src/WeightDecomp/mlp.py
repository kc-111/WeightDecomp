import torch
import torch.nn as nn
import torch.nn.functional as F

from .decomposed_linear import DecomposedLinear


class DecomposedMLP(nn.Module):
    """MLP with ReLU activations using DecomposedLinear layers.

    Each layer's weight is parameterized as W_eff = W + sum_i B_i @ C_i,
    enabling dead neuron revival through the coupling matrix P = I + sum B_i B_i^T.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        ranks_per_layer: list[list[int]] | None = None,
        alpha: float = 1.0,
    ):
        """
        Args:
            layer_sizes: [input_dim, hidden1, hidden2, ..., output_dim].
            ranks_per_layer: list of rank lists, one per layer. Length must be
                len(layer_sizes) - 1. None means no factors on any layer.
            alpha: scaling on the base weight W.
        """
        super().__init__()
        n_layers = len(layer_sizes) - 1
        if ranks_per_layer is not None and len(ranks_per_layer) != n_layers:
            raise ValueError(
                f"ranks_per_layer has {len(ranks_per_layer)} entries "
                f"but there are {n_layers} layers"
            )

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            ranks = ranks_per_layer[i] if ranks_per_layer else None
            self.layers.append(
                DecomposedLinear(layer_sizes[i], layer_sizes[i + 1],
                                ranks=ranks, alpha=alpha)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

    def merge_all(self, rerandomize_B: bool = True) -> None:
        for layer in self.layers:
            layer.merge(rerandomize_B=rerandomize_B)

    def split_all(self, ranks: list[int], rerandomize_B: bool = True) -> None:
        for layer in self.layers:
            layer.split(ranks, rerandomize_B=rerandomize_B)

    def decomposed_layers(self) -> list[DecomposedLinear]:
        return list(self.layers)

    def grow_width(self, new_hidden: int, init_scale: float = 0.1) -> None:
        """Grow all hidden layers to new_hidden width.

        Input/output dims are preserved. Each hidden layer grows its output,
        and the next layer grows its input to match.
        """
        for i, layer in enumerate(self.layers):
            is_first = i == 0
            is_last = i == len(self.layers) - 1

            new_out = layer.out_features if is_last else new_hidden
            new_in = layer.in_features if is_first else new_hidden

            if new_out != layer.out_features or new_in != layer.in_features:
                layer.grow(new_out=new_out, new_in=new_in, init_scale=init_scale)
