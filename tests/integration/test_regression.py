"""
Integration test: PhotoKAN learns a simple symbolic function.
y = sin(x1) + x2^2 on CPU simulation.
"""

import pytest
import torch

from photokan.layers import PhotoKAN


@pytest.mark.slow
def test_regression_converges():
    torch.manual_seed(0)
    x = torch.rand(400, 2) * 4 - 2
    y = torch.sin(x[:, 0]) + x[:, 1] ** 2

    x_tr, y_tr = x[:300], y[:300]
    x_te, y_te = x[300:], y[300:]

    model = PhotoKAN(
        layer_sizes=[2, 8, 1],
        activation="sine",
        backend="cpu",
        noise_sim=False,
        n_basis=6,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    for _ in range(200):
        pred = model(x_tr).squeeze()
        loss = torch.nn.functional.mse_loss(pred, y_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_loss = torch.nn.functional.mse_loss(model(x_te).squeeze(), y_te).item()

    # Should achieve reasonable fit in 200 steps
    assert test_loss < 1.0, f"Test MSE {test_loss:.4f} too high"


def test_regression_smoke():
    """Quick smoke test — just checks training runs without errors."""
    torch.manual_seed(42)
    x = torch.rand(32, 2)
    y = x[:, 0] + x[:, 1]

    model = PhotoKAN(
        layer_sizes=[2, 4, 1],
        activation="relu",
        backend="cpu",
        noise_sim=False,
        n_basis=4,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(5):
        pred = model(x).squeeze()
        loss = torch.nn.functional.mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 100  # just no crash
