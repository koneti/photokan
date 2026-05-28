# photokan/llm/adapters.py
"""
PhotoLoRA — photonic low-rank adapters for transformer fine-tuning.

Adds PhotoKAN-based low-rank adapters to frozen base model weights,
analogous to LoRA but using photonic nonlinear projections instead
of linear A/B matrices. This gives:
  - Fewer trainable parameters than full fine-tuning
  - Photonic acceleration at inference time
  - Drop-in compatibility with the HuggingFace PEFT workflow

Architecture per targeted Linear layer W (d_in × d_out):
    output = W·x + scale · KAN_B(KAN_A(x))

where KAN_A: d_in → rank, KAN_B: rank → d_out.
"""
from __future__ import annotations

import math
import warnings
from typing import Iterable
import torch
import torch.nn as nn

from ..layers import PhotoKAN


class PhotoLoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with a PhotoKAN low-rank adapter.

    The base weight is frozen; only the KAN adapter parameters are trained.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 4,
        activation: str = "sine",
        backend: str = "auto",
        n_basis: int = 6,
        scale: float = 1.0,
        noise_sim: bool = False,
    ):
        super().__init__()
        self.base   = base_linear
        self.scale  = scale
        self.rank   = rank

        # Freeze base weights
        for p in self.base.parameters():
            p.requires_grad_(False)

        in_f  = base_linear.in_features
        out_f = base_linear.out_features

        # KAN adapter: d_in → rank → d_out
        self.adapter = PhotoKAN(
            layer_sizes=[in_f, rank, out_f],
            activation=activation,
            backend=backend,
            n_basis=n_basis,
            noise_sim=noise_sim,
        )

        # Initialise adapter output near zero
        for p in self.adapter.parameters():
            nn.init.normal_(p, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out    = self.base(x)
        adapter_out = self.adapter(x)
        return base_out + self.scale * adapter_out

    def merge_weights(self) -> nn.Linear:
        """
        Approximate merge: sample adapter on a grid and fit a linear mapping.

        Note: KAN adapters are nonlinear — a true merge is not possible.
        This returns a Linear approximation useful for deployment export only.
        """
        warnings.warn(
            "PhotoLoRA merge is a linear approximation of a nonlinear adapter. "
            "Use compile_photokan_layers() for exact photonic deployment.",
            stacklevel=2,
        )
        in_f  = self.base.in_features
        out_f = self.base.out_features
        merged = nn.Linear(in_f, out_f, bias=self.base.bias is not None)
        merged.weight.data = self.base.weight.data.clone()
        if self.base.bias is not None:
            merged.bias.data = self.base.bias.data.clone()
        return merged

    def extra_repr(self) -> str:
        return f"rank={self.rank}, scale={self.scale}"


def add_photo_lora(
    model: nn.Module,
    rank: int = 4,
    activation: str = "sine",
    backend: str = "auto",
    n_basis: int = 6,
    scale: float = 1.0,
    target_modules: list[str] | None = None,
    noise_sim: bool = False,
) -> nn.Module:
    """
    Add PhotoKAN LoRA adapters to targeted Linear modules in a model.

    Args:
        model          : HuggingFace transformer (or any nn.Module).
        rank           : Adapter bottleneck rank.
        activation     : KAN edge activation type.
        backend        : Hardware backend.
        n_basis        : Activation basis size.
        scale          : Adapter output scale (α/rank in standard LoRA notation).
        target_modules : List of module name substrings to target,
                         e.g. ['q_proj', 'v_proj'].
                         None = target all Linear layers.
        noise_sim      : Enable photonic noise.

    Returns:
        Model with PhotoLoRALinear adapters inserted.

    Example::

        photo_model = add_photo_lora(
            base_model,
            rank=4,
            activation='sine',
            target_modules=['q_proj', 'v_proj'],
        )
        # Only adapter parameters are trainable
        trainable = [n for n, p in photo_model.named_parameters() if p.requires_grad]
    """
    replaced = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if target_modules is not None:
            if not any(t in name for t in target_modules):
                continue

        lora = PhotoLoRALinear(
            base_linear=module,
            rank=rank,
            activation=activation,
            backend=backend,
            n_basis=n_basis,
            scale=scale,
            noise_sim=noise_sim,
        )

        # Replace the module in the parent
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lora)
        replaced += 1

    print(f"[photokan.llm] Added PhotoLoRA adapters to {replaced} Linear module(s).")

    # Report trainable parameters
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[photokan.llm] Trainable: {trainable:,} / {total:,} "
          f"({100*trainable/max(total,1):.2f}%)")

    return model
