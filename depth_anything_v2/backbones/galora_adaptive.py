"""
Gated LoRA with layer-adaptive rank for encoder fine-tuning.

Two ideas combined:
1. Gated LoRA: the low-rank update is modulated by a sigmoid self-gate so that
   the encoder can learn *when* to apply the domain shift instead of always
   adding it. Mitigates over-adaptation on small datasets.
2. Layer-adaptive rank: shallow blocks (general low-level features) keep a
   small rank, deep blocks (domain semantics) get a larger rank. Linear
   interpolation between rank_min and rank_max across encoder depth.
"""

import math
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLoRALinear(nn.Module):
    """Wrap an existing nn.Linear with a gated low-rank update.

    update = sigmoid(delta) * delta,   delta = (dropout(x) @ A^T @ B^T) * scaling
    out    = base_linear(x) + update

    The no_seed_ori weight/bias are frozen; only lora_A and lora_B are trained.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int,
        lora_alpha: float,
        lora_dropout: float,
    ) -> None:
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = int(r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / self.r if self.r > 0 else 0.0

        # Reuse the original parameters and freeze them.
        self.weight = base_linear.weight
        self.bias = base_linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        self.lora_dropout = (
            nn.Dropout(p=lora_dropout) if lora_dropout and lora_dropout > 0.0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            delta = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t() * self.scaling
            gate = torch.sigmoid(delta)
            result = result + delta * gate
        return result

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.lora_alpha}"
        )


def get_layer_ranks(num_layers: int, rank_min: int, rank_max: int) -> List[int]:
    """Linearly interpolate ranks across encoder depth."""
    if num_layers <= 1:
        return [int(rank_max)]
    ranks = []
    for i in range(num_layers):
        r = rank_min + (rank_max - rank_min) * i / (num_layers - 1)
        ranks.append(int(round(r)))
    return ranks


def _get_submodule(root: nn.Module, dotted: str):
    parent = root
    parts = dotted.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_gated_lora(
    encoder: nn.Module,
    rank_min: int = 2,
    rank_max: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.1,
    target_modules: Iterable[str] = ("attn.qkv", "mlp.fc1", "mlp.fc2"),
) -> List[int]:
    """Inject GatedLoRALinear into every transformer block of `encoder`.

    Walks `encoder.blocks`, replaces each `target_modules` entry with a
    GatedLoRALinear of the rank prescribed for that layer, and freezes every
    non-LoRA parameter inside the encoder.

    Returns the per-layer rank list (useful for logging).
    """
    blocks = encoder.blocks
    num_layers = len(blocks)
    ranks = get_layer_ranks(num_layers, rank_min, rank_max)

    for layer_idx, block in enumerate(blocks):
        r = ranks[layer_idx]
        for target in target_modules:
            parent, attr_name = _get_submodule(block, target)
            base_linear = getattr(parent, attr_name)
            if not isinstance(base_linear, nn.Linear):
                raise TypeError(
                    f"target {target} on block {layer_idx} is {type(base_linear).__name__}, "
                    f"expected nn.Linear"
                )
            wrapped = GatedLoRALinear(
                base_linear=base_linear,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            setattr(parent, attr_name, wrapped)

    # Freeze every non-LoRA parameter in the encoder.
    for name, p in encoder.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    return ranks
