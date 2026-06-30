from __future__ import annotations
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dinov3.utils import cat_keep_shapes, uncat_with_shapes


class ListForwardMixin(object):
    def forward(self, x: Tensor):
        raise NotImplementedError

    def forward_list(self, x_list: List[Tensor]) -> List[Tensor]:
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        x_flat = self.forward(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)


class Mlp(nn.Module, ListForwardMixin):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device)
        self.drop = nn.Dropout(drop)
        self.lora_rank = 0
        self._lora_scale = 1.0

    def enable_lora(self, rank: int, n_experts: int = 2) -> None:
        self.lora_rank = rank
        self.n_experts = n_experts
        dim = self.fc1.in_features
        hidden = self.fc1.out_features
        out = self.fc2.out_features
        # fc1 MoE LoRA
        self.fc1_lora_As = nn.ModuleList([nn.Linear(dim, rank, bias=False) for _ in range(n_experts)])
        self.fc1_lora_Bs = nn.ModuleList([nn.Linear(rank, hidden, bias=False) for _ in range(n_experts)])
        self.fc1_lora_gate = nn.Linear(dim, n_experts, bias=False)
        for lora_A in self.fc1_lora_As:
            nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
        for lora_B in self.fc1_lora_Bs:
            nn.init.normal_(lora_B.weight, std=0.01)
        nn.init.zeros_(self.fc1_lora_gate.weight)
        # fc2 MoE LoRA
        self.fc2_lora_As = nn.ModuleList([nn.Linear(hidden, rank, bias=False) for _ in range(n_experts)])
        self.fc2_lora_Bs = nn.ModuleList([nn.Linear(rank, out, bias=False) for _ in range(n_experts)])
        self.fc2_lora_gate = nn.Linear(dim, n_experts, bias=False)
        for lora_A in self.fc2_lora_As:
            nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
        for lora_B in self.fc2_lora_Bs:
            nn.init.normal_(lora_B.weight, std=0.01)
        nn.init.zeros_(self.fc2_lora_gate.weight)

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc1(x)
        if self.lora_rank > 0:
            scale = self._lora_scale
            if isinstance(scale, torch.Tensor):
                if scale.dim() == 1:
                    scale = scale.view(scale.shape[0], 1, 1)
                else:
                    scale = scale.unsqueeze(-1)
            g1 = torch.softmax(self.fc1_lora_gate(x.mean(dim=1)), dim=-1)  # [B, n_experts]
            delta1 = sum(g1[:, k].view(-1, 1, 1) * self.fc1_lora_Bs[k](self.fc1_lora_As[k](x))
                         for k in range(self.n_experts))
            h = h + scale * delta1
        h = self.act(h)
        h = self.drop(h)
        out = self.fc2(h)
        if self.lora_rank > 0:
            scale = self._lora_scale
            if isinstance(scale, torch.Tensor):
                if scale.dim() == 1:
                    scale = scale.view(scale.shape[0], 1, 1)
                else:
                    scale = scale.unsqueeze(-1)
            g2 = torch.softmax(self.fc2_lora_gate(x.mean(dim=1)), dim=-1)  # [B, n_experts]
            delta2 = sum(g2[:, k].view(-1, 1, 1) * self.fc2_lora_Bs[k](self.fc2_lora_As[k](h))
                         for k in range(self.n_experts))
            out = out + scale * delta2
        out = self.drop(out)
        return out


class SwiGLUFFN(nn.Module, ListForwardMixin):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[Callable[..., nn.Module]] = None,
        drop: float = 0.0,
        bias: bool = True,
        align_to: int = 8,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % align_to)
        self.w1 = nn.Linear(in_features, swiglu_hidden_features, bias=bias, device=device)
        self.w2 = nn.Linear(in_features, swiglu_hidden_features, bias=bias, device=device)
        self.w3 = nn.Linear(swiglu_hidden_features, out_features, bias=bias, device=device)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)
