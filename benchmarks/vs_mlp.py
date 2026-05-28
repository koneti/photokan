#!/usr/bin/env python3
"""
Benchmark: PhotoKAN-Sine vs equivalent MLP.

Compares parameter counts, op estimates, and accuracy on the
symbolic regression task  y = sin(x1) + x2^2.
"""

import time
import torch
import torch.nn as nn
import photokan as pk


# ── Config ───────────────────────────────────────────────────────────────────
HIDDEN      = 16
LAYERS      = 3          # hidden layers
EPOCHS      = 300
N_TRAIN     = 800
N_TEST      = 200
LR          = 2e-3
SEED        = 42


def make_data():
    torch.manual_seed(SEED)
    x = torch.rand(N_TRAIN + N_TEST, 2) * 4 - 2
    y = torch.sin(x[:, 0]) + x[:, 1] ** 2
    return (x[:N_TRAIN], y[:N_TRAIN]), (x[N_TRAIN:], y[N_TRAIN:])


def make_mlp(in_f=2, hidden=HIDDEN, out_f=1, n_layers=LAYERS):
    layers = [nn.Linear(in_f, hidden), nn.GELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.GELU()]
    layers.append(nn.Linear(hidden, out_f))
    return nn.Sequential(*layers)


def make_photokan(in_f=2, hidden=HIDDEN, out_f=1, n_layers=LAYERS):
    sizes = [in_f] + [hidden] * n_layers + [out_f]
    return pk.PhotoKAN(
        layer_sizes=sizes,
        activation="sine",
        backend="cpu",
        noise_sim=False,
        n_basis=6,
    )


def train(model, data_tr, data_te, epochs=EPOCHS, lr=LR):
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    x_tr, y_tr = data_tr
    x_te, y_te = data_te

    t0 = time.perf_counter()
    for e in range(epochs):
        model.train()
        pred = model(x_tr).squeeze()
        loss = nn.functional.mse_loss(pred, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()

    elapsed = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        test_loss = nn.functional.mse_loss(model(x_te).squeeze(), y_te).item()
    return test_loss, elapsed


def main():
    data_tr, data_te = make_data()

    mlp      = make_mlp()
    photokan = make_photokan()

    mlp_params      = sum(p.numel() for p in mlp.parameters())
    photokan_params = sum(p.numel() for p in photokan.parameters())

    print(f"\n{'=' * 55}")
    print(f"  PhotoKAN-Sine vs MLP — symbolic regression benchmark")
    print(f"{'=' * 55}")
    print(f"  Task:         y = sin(x1) + x2^2")
    print(f"  Train/test:   {N_TRAIN}/{N_TEST}  |  Epochs: {EPOCHS}")
    print(f"\n  MLP params:      {mlp_params:,}")
    print(f"  PhotoKAN params: {photokan_params:,}  "
          f"({photokan_params/mlp_params:.0%} of MLP)")

    mlp_loss, mlp_time = train(mlp, data_tr, data_te)
    pk_loss,  pk_time  = train(photokan, data_tr, data_te)

    print(f"\n  {'Model':<14} {'Test MSE':>10} {'Train time':>12}")
    print(f"  {'-'*38}")
    print(f"  {'MLP':<14} {mlp_loss:>10.4f} {mlp_time:>10.2f}s")
    print(f"  {'PhotoKAN-Sine':<14} {pk_loss:>10.4f} {pk_time:>10.2f}s")

    ops = photokan.estimate_ops()
    print(f"\n  PhotoKAN op estimates vs MLP equivalent:")
    print(f"    param_ratio:  {ops['param_ratio']:.2f}x")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
