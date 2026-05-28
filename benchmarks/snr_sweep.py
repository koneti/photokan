#!/usr/bin/env python3
"""Benchmark: model accuracy across SNR levels."""

import torch
import photokan as pk


def main():
    torch.manual_seed(0)
    x = torch.rand(200, 2) * 4 - 2
    y = torch.sin(x[:, 0]) + x[:, 1] ** 2

    model = pk.PhotoKAN(
        layer_sizes=[2, 8, 1],
        activation="sine",
        backend="cpu",
        noise_sim=False,
        n_basis=6,
    )
    # Quick train
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(200):
        loss = torch.nn.functional.mse_loss(model(x).squeeze(), y)
        opt.zero_grad(); loss.backward(); opt.step()

    sim = pk.PhotonicSimulator()
    results = sim.sweep_snr(model, x, y, snr_range=[8, 10, 12, 14, 16, 20])

    print(f"\n{'SNR (dB)':>10}  {'MSE':>12}")
    print(f"  {'-' * 24}")
    for snr, score in zip(results["snr_db"], results["scores"]):
        print(f"  {snr:>8.0f}  {score:>12.4f}")
    print()


if __name__ == "__main__":
    main()
