"""
PAS-LoRA: patch-adaptive-scale MoE LoRA for DINOv3 ViT.

Collected from:
  - dinov3/layers/attention.py     SelfAttention.enable_lora / LoRA branch of forward
  - dinov3/layers/ffn_layers.py    Mlp.enable_lora / LoRA branch of forward
  - dinov3/models/vision_transformer.py
        DynamicStorageAdapter, StorageGuidedLoRAScaleGenerator,
        enable_dynamic_lora, per-block scale generation in
        _get_intermediate_layers_not_chunked
  - train.py                       freezing backbone / unfreezing LoRA params

Attribute names (qkv_lora_As, fc1_lora_gate, lora_scale_gen, storage_adapter, ...)
are identical to the original, so existing checkpoints load unchanged.
"""
import math

import torch
import torch.nn as nn
from torch import Tensor


# ============================================================
# Scale / storage modules
# ============================================================
class DynamicStorageAdapter(nn.Module):
    """Image-conditioned offset for the DINOv3 storage (register) tokens."""
    def __init__(self, embed_dim: int, n_storage_tokens: int):
        super().__init__()
        self.n = n_storage_tokens
        self.d = embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, n_storage_tokens * embed_dim),
        )

    def forward(self, patch_embeds: Tensor) -> Tensor:
        g = patch_embeds.mean(dim=1)                      # [B, C]
        return self.net(g).reshape(-1, self.n, self.d)    # [B, n, C]


class StorageGuidedLoRAScaleGenerator(nn.Module):
    """Generates per-token per-block LoRA scales (patch-adaptive scaling)."""
    def __init__(self, embed_dim: int, n_blocks: int):
        super().__init__()
        self.n_blocks = n_blocks
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, n_blocks),
            nn.Sigmoid(),
        )

    def forward(self, patch_tokens: Tensor) -> Tensor:
        # patch_tokens: [B, N, C] -> [B, N, n_blocks]
        return self.net(patch_tokens)


# ============================================================
# MoE LoRA core
# ============================================================
def _make_moe_lora(in_dim: int, out_dim: int, rank: int, n_experts: int, gate_dim: int):
    """Build (As, Bs, gate) for one MoE-LoRA branch with the original init."""
    As = nn.ModuleList([nn.Linear(in_dim, rank, bias=False) for _ in range(n_experts)])
    Bs = nn.ModuleList([nn.Linear(rank, out_dim, bias=False) for _ in range(n_experts)])
    gate = nn.Linear(gate_dim, n_experts, bias=False)
    for lora_A in As:
        nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
    for lora_B in Bs:
        nn.init.normal_(lora_B.weight, std=0.01)
    nn.init.zeros_(gate.weight)
    return As, Bs, gate


def _reshape_scale(scale):
    """float -> float; [B] -> [B,1,1]; [B,N] -> [B,N,1]."""
    if isinstance(scale, torch.Tensor):
        if scale.dim() == 1:
            return scale.view(scale.shape[0], 1, 1)
        return scale.unsqueeze(-1)
    return scale


def moe_lora_delta(inp: Tensor, gate_in: Tensor, As, Bs, gate, scale) -> Tensor:
    """scale * sum_k softmax(gate(mean(gate_in)))_k * B_k(A_k(inp))"""
    g = torch.softmax(gate(gate_in.mean(dim=1)), dim=-1)   # [B, n_experts]
    delta = sum(g[:, k].view(-1, 1, 1) * Bs[k](As[k](inp)) for k in range(len(As)))
    return _reshape_scale(scale) * delta


# ============================================================
# Attach LoRA to attention / MLP
# ============================================================
def enable_attn_lora(attn: nn.Module, rank: int, n_experts: int = 2) -> None:
    """Add MoE LoRA on attn.qkv (same as SelfAttention.enable_lora)."""
    attn.lora_rank = rank
    attn.n_experts = n_experts
    dim = attn.qkv.in_features
    attn.qkv_lora_As, attn.qkv_lora_Bs, attn.qkv_lora_gate = _make_moe_lora(dim, dim * 3, rank, n_experts, dim)


def attn_qkv_with_lora(attn: nn.Module, x: Tensor) -> Tensor:
    """qkv projection used in SelfAttention.forward."""
    qkv = attn.qkv(x)
    if getattr(attn, 'lora_rank', 0) > 0:
        qkv = qkv + moe_lora_delta(x, x, attn.qkv_lora_As, attn.qkv_lora_Bs, attn.qkv_lora_gate, attn._lora_scale)
    return qkv


def enable_mlp_lora(mlp: nn.Module, rank: int, n_experts: int = 2) -> None:
    """Add MoE LoRA on mlp.fc1 and mlp.fc2 (same as Mlp.enable_lora)."""
    mlp.lora_rank = rank
    mlp.n_experts = n_experts
    dim = mlp.fc1.in_features
    hidden = mlp.fc1.out_features
    out = mlp.fc2.out_features
    mlp.fc1_lora_As, mlp.fc1_lora_Bs, mlp.fc1_lora_gate = _make_moe_lora(dim, hidden, rank, n_experts, dim)
    # fc2 gate is driven by the block input x (dim), not by the hidden h
    mlp.fc2_lora_As, mlp.fc2_lora_Bs, mlp.fc2_lora_gate = _make_moe_lora(hidden, out, rank, n_experts, dim)


def mlp_forward_with_lora(mlp: nn.Module, x: Tensor) -> Tensor:
    """Same as Mlp.forward."""
    use_lora = getattr(mlp, 'lora_rank', 0) > 0
    h = mlp.fc1(x)
    if use_lora:
        h = h + moe_lora_delta(x, x, mlp.fc1_lora_As, mlp.fc1_lora_Bs, mlp.fc1_lora_gate, mlp._lora_scale)
    h = mlp.act(h)
    h = mlp.drop(h)
    out = mlp.fc2(h)
    if use_lora:
        out = out + moe_lora_delta(h, x, mlp.fc2_lora_As, mlp.fc2_lora_Bs, mlp.fc2_lora_gate, mlp._lora_scale)
    return mlp.drop(out)


# ============================================================
# ViT-level
# ============================================================
def enable_dynamic_lora(vit: nn.Module, rank: int = 8, n_experts: int = 2) -> None:
    """Same as DinoVisionTransformer.enable_dynamic_lora."""
    if vit.n_storage_tokens > 0:
        vit.storage_adapter = DynamicStorageAdapter(vit.embed_dim, vit.n_storage_tokens)
    for blk in vit.blocks:
        enable_attn_lora(blk.attn, rank, n_experts)
        enable_mlp_lora(blk.mlp, rank, n_experts)
    vit.lora_scale_gen = StorageGuidedLoRAScaleGenerator(vit.embed_dim, vit.n_blocks)


def compute_lora_scales(vit: nn.Module, tokens: Tensor) -> Tensor:
    """tokens: [B, 1 + n_storage + N_patch, C] after prepare_tokens_with_masks.
    Returns [B, N_total, n_blocks]; prefix tokens (cls + storage) get scale 1."""
    n_prefix = 1 + vit.n_storage_tokens
    patch_scales = vit.lora_scale_gen(tokens[:, n_prefix:])            # [B, N_patch, n_blocks]
    B = tokens.shape[0]
    prefix_scales = torch.ones(B, n_prefix, vit.lora_scale_gen.n_blocks,
                               device=tokens.device, dtype=patch_scales.dtype)
    return torch.cat([prefix_scales, patch_scales], dim=1)


def set_block_lora_scale(blk: nn.Module, lora_scales: Tensor, i: int) -> None:
    """Assign block i's scale [B, N_total] before calling blk(x, rope)."""
    blk.attn._lora_scale = lora_scales[:, :, i]
    blk.mlp._lora_scale = lora_scales[:, :, i]


# ============================================================
# Training helper (from train.py)
# ============================================================
def mark_only_lora_trainable(vit: nn.Module) -> None:
    """Freeze the backbone, then unfreeze LoRA A/B/gate and lora_scale_gen."""
    for p in vit.parameters():
        p.requires_grad = False
    lora_modules = (
        [blk.attn.qkv_lora_As for blk in vit.blocks] +
        [blk.attn.qkv_lora_Bs for blk in vit.blocks] +
        [blk.attn.qkv_lora_gate for blk in vit.blocks] +
        [blk.mlp.fc1_lora_As for blk in vit.blocks] +
        [blk.mlp.fc1_lora_Bs for blk in vit.blocks] +
        [blk.mlp.fc1_lora_gate for blk in vit.blocks] +
        [blk.mlp.fc2_lora_As for blk in vit.blocks] +
        [blk.mlp.fc2_lora_Bs for blk in vit.blocks] +
        [blk.mlp.fc2_lora_gate for blk in vit.blocks] +
        [vit.lora_scale_gen]
    )
    for m in lora_modules:
        for p in m.parameters():
            p.requires_grad = True
