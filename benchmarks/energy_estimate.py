#!/usr/bin/env python3
"""Benchmark: photonic vs CMOS energy estimates."""

import photokan as pk
from photokan.utils.energy import estimate_model_energy


def main():
    model = pk.PhotoKAN(
        layer_sizes=[4, 16, 16, 1],
        activation="sine",
        backend="cpu",
        noise_sim=False,
        n_basis=8,
    )

    print("\nPhotoKAN Energy Estimates (batch_size=1)")
    print("=" * 50)
    reports = estimate_model_energy(model, batch_size=1)
    for i, r in enumerate(reports):
        print(f"\nLayer {i}: {r.summary()}")
    print()


if __name__ == "__main__":
    main()
