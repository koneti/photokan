# photokan/utils/energy.py
"""
Energy cost estimator for photonic KAN workloads.

Estimates energy consumption based on Q.ANT published figures:
- TFLN photonic MAC: ~30x more efficient than CMOS
- CMOS 40nm MAC: ~3.7 pJ/op (industry reference)
- Photonic MAC: ~0.12 pJ/op (Q.ANT claim)

Energy per operation varies by activation type:
- Sine/Fourier: single optical pass through waveguide (lowest energy)
- Spline: requires LUT lookup (slightly higher)
- ReLU: piecewise linear, minimal ops
"""

from __future__ import annotations

from dataclasses import dataclass

# Energy per MAC operation in picojoules
_CMOS_MAC_PJ = 3.7        # 40nm CMOS reference
_PHOTONIC_MAC_PJ = 0.12   # Q.ANT TFLN claim (~30x improvement)

# Overhead multipliers per activation type
_ACTIVATION_ENERGY: dict[str, float] = {
    "sine": 1.0,       # Direct optical — baseline
    "fourier": 1.2,    # Extra cosine component
    "spline": 1.5,     # LUT access overhead
    "relu": 0.8,       # Simplest op
}


@dataclass
class EnergyReport:
    """Energy estimation results."""
    n_edges: int
    n_ops_per_edge: int
    total_ops: int
    cmos_energy_uj: float
    photonic_energy_uj: float
    efficiency_ratio: float
    activation_type: str

    def summary(self) -> str:
        return (
            f"Energy Report ({self.activation_type})\n"
            f"  Edges: {self.n_edges:,}  |  Ops/edge: {self.n_ops_per_edge}\n"
            f"  Total ops: {self.total_ops:,}\n"
            f"  CMOS energy:     {self.cmos_energy_uj:.3f} uJ\n"
            f"  Photonic energy: {self.photonic_energy_uj:.4f} uJ\n"
            f"  Efficiency:      {self.efficiency_ratio:.1f}x better than CMOS"
        )


def estimate_layer_energy(
    layer,
    batch_size: int = 1,
    cmos_mac_pj: float = _CMOS_MAC_PJ,
    photonic_mac_pj: float = _PHOTONIC_MAC_PJ,
) -> EnergyReport:
    """
    Estimate energy consumption for a single PhotoKANLayer forward pass.

    Args:
        layer         : PhotoKANLayer instance.
        batch_size    : Number of samples in the batch.
        cmos_mac_pj   : Energy per CMOS MAC in picojoules.
        photonic_mac_pj: Energy per photonic MAC in picojoules.

    Returns:
        EnergyReport with detailed breakdown.
    """
    n_edges = layer.in_features * layer.out_features

    # Each edge activation involves n_basis multiply-accumulate ops
    # (sin components, spline coefficients, etc.)
    activation = layer.edge_activations[0]
    act_type = type(activation).__name__.lower().replace("edgeactivation", "")

    # Count ops per edge from parameter count
    n_params = sum(p.numel() for p in activation.parameters())
    n_ops_per_edge = max(n_params, 1)

    total_ops = n_edges * n_ops_per_edge * batch_size

    # Apply activation-specific overhead
    overhead = _ACTIVATION_ENERGY.get(act_type, 1.0)

    cmos_energy_uj = total_ops * cmos_mac_pj * overhead * 1e-6
    photonic_energy_uj = total_ops * photonic_mac_pj * overhead * 1e-6
    efficiency_ratio = cmos_energy_uj / max(photonic_energy_uj, 1e-15)

    return EnergyReport(
        n_edges=n_edges,
        n_ops_per_edge=n_ops_per_edge,
        total_ops=total_ops,
        cmos_energy_uj=cmos_energy_uj,
        photonic_energy_uj=photonic_energy_uj,
        efficiency_ratio=efficiency_ratio,
        activation_type=act_type,
    )


def estimate_model_energy(
    model,
    batch_size: int = 1,
    cmos_mac_pj: float = _CMOS_MAC_PJ,
    photonic_mac_pj: float = _PHOTONIC_MAC_PJ,
) -> list[EnergyReport]:
    """
    Estimate energy for all layers in a PhotoKAN model.

    Args:
        model         : PhotoKAN instance.
        batch_size    : Number of samples in the batch.
        cmos_mac_pj   : Energy per CMOS MAC in picojoules.
        photonic_mac_pj: Energy per photonic MAC in picojoules.

    Returns:
        List of EnergyReport, one per layer.
    """
    reports = []
    for layer in model.layers:
        report = estimate_layer_energy(layer, batch_size, cmos_mac_pj, photonic_mac_pj)
        reports.append(report)

    total_cmos = sum(r.cmos_energy_uj for r in reports)
    total_photonic = sum(r.photonic_energy_uj for r in reports)
    print(f"Total CMOS:     {total_cmos:.3f} uJ")
    print(f"Total Photonic: {total_photonic:.4f} uJ")
    print(f"Overall ratio:  {total_cmos / max(total_photonic, 1e-15):.1f}x")

    return reports
