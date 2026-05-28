# photokan/llm/replacer.py
"""
LLM integration — replace MLP/FFN layers with PhotoKAN.

Supports HuggingFace transformer models. Identifies feed-forward
sublayers (GPT-2, LLaMA, Mistral, Falcon patterns) and swaps them
out with PhotoKANLayer stacks, preserving residual connections and
model interfaces.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from ..layers import PhotoKAN

# ---------------------------------------------------------------------------
# Known FFN module patterns per model family
# ---------------------------------------------------------------------------


def _get_mlp_modules(model) -> list[tuple[str, nn.Module]]:
    """
    Return a list of (name, module) pairs for all MLP/FFN sublayers
    in a HuggingFace transformer model.

    Detects: GPT-2 MLP, LLaMA MLP, Mistral MLP, Falcon MLP, generic Linear pairs.
    """
    candidates = []
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        # HuggingFace naming conventions
        if cls_name in (
            "GPT2MLP",
            "LlamaMLP",
            "MistralMLP",
            "FalconMLP",
            "MistralMLP",
            "PhiMLP",
            "GemmaMLP",
        ):
            candidates.append((name, module))
        elif "mlp" in name.lower() and isinstance(module, nn.Module):
            # Generic: any module named *mlp* that contains linear layers
            has_linear = any(isinstance(m, nn.Linear) for m in module.children())
            if has_linear and name not in [n for n, _ in candidates]:
                candidates.append((name, module))
    return candidates


def _infer_mlp_dims(mlp_module: nn.Module) -> tuple[int, int]:
    """Infer input and output dimensions from an MLP module's linear layers."""
    linears = [m for m in mlp_module.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError(f"No Linear layers found in {type(mlp_module).__name__}")
    in_features = linears[0].in_features
    out_features = linears[-1].out_features
    return in_features, out_features


def _set_nested_attr(obj, attr_path: str, value):
    """Set a nested attribute by dot-separated path."""
    parts = attr_path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def replace_mlp_with_photokan(
    model: nn.Module,
    activation: str = "sine",
    backend: str = "auto",
    n_basis: int = 8,
    layers_to_replace: list[int] | None = None,
    preserve_residuals: bool = True,
    noise_sim: bool = False,
    hidden_sizes: list[int] | None = None,
) -> nn.Module:
    """
    Replace MLP/FFN sublayers in a transformer model with PhotoKAN layers.

    Args:
        model             : HuggingFace transformer model.
        activation        : KAN edge activation type.
        backend           : Hardware backend.
        n_basis           : Activation basis size.
        layers_to_replace : List of transformer block indices to replace.
                            None = replace all detected MLP modules.
        preserve_residuals: If True, wrap the PhotoKAN in a residual block
                            that adds the original input (requires in==out dims).
        noise_sim         : Enable photonic noise simulation.
        hidden_sizes      : Custom hidden layer sizes for PhotoKAN replacement.
                            If None, uses [in_f, max(in_f, out_f), out_f].

    Returns:
        Model with PhotoKAN layers substituted in-place.

    Example::

        from transformers import GPT2LMHeadModel
        import photokan.llm as pkl

        base = GPT2LMHeadModel.from_pretrained('gpt2')
        photo_model = pkl.replace_mlp_with_photokan(
            base, activation='sine', layers_to_replace=[0, 1, 2]
        )
    """
    mlp_modules = _get_mlp_modules(model)

    if not mlp_modules:
        warnings.warn(
            "No MLP/FFN modules detected. Supported: GPT-2, LLaMA, Mistral, Falcon. "
            "For custom architectures, pass modules manually.",
            stacklevel=2,
        )
        return model

    # Filter by layer index if specified
    if layers_to_replace is not None:
        mlp_modules = [
            (name, mod) for i, (name, mod) in enumerate(mlp_modules) if i in layers_to_replace
        ]

    replaced = 0
    for name, mlp_mod in mlp_modules:
        try:
            in_f, out_f = _infer_mlp_dims(mlp_mod)
        except ValueError as e:
            warnings.warn(f"Skipping {name}: {e}", stacklevel=2)
            continue

        if hidden_sizes is None:
            hidden = max(in_f, out_f)
            sizes = [in_f, hidden, out_f]
        else:
            sizes = [in_f] + list(hidden_sizes) + [out_f]

        photokan = PhotoKAN(
            layer_sizes=sizes,
            activation=activation,
            backend=backend,
            n_basis=n_basis,
            noise_sim=noise_sim,
        )

        if preserve_residuals and in_f == out_f:
            replacement = _ResidualPhotoKAN(photokan)
        else:
            replacement = photokan

        _set_nested_attr(model, name, replacement)
        replaced += 1

    print(f"[photokan.llm] Replaced {replaced} MLP module(s) with PhotoKAN.")
    return model


class _ResidualPhotoKAN(nn.Module):
    """PhotoKAN with skip connection (input + KAN(input))."""

    def __init__(self, photokan: PhotoKAN):
        super().__init__()
        self.kan = photokan

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.kan(x)


def compile_photokan_layers(
    model: nn.Module,
    output_path: str,
    **compiler_kwargs,
) -> dict:
    """
    Compile all PhotoKAN layers in a model to .npu bundles.

    Each PhotoKAN layer gets its own numbered bundle under output_path/.

    Args:
        model       : Model containing PhotoKAN or PhotoKANLayer instances.
        output_path : Directory to write bundles.
        **compiler_kwargs: Forwarded to PhotonicCompiler.

    Returns:
        Dict mapping layer name → bundle path.
    """
    from ..compiler import PhotonicCompiler

    compiler = PhotonicCompiler(**compiler_kwargs)
    bundles = {}

    for name, module in model.named_modules():
        if isinstance(module, PhotoKAN):
            bundle_path = f"{output_path}/{name.replace('.', '_')}.npu"
            try:
                compiler.compile(module, bundle_path, validate=False)
                bundles[name] = bundle_path
                print(f"[photokan.llm] Compiled {name} → {bundle_path}")
            except Exception as e:
                warnings.warn(f"Failed to compile {name}: {e}", stacklevel=2)

    return bundles
