#!/usr/bin/env python3
"""
PhotoKAN as a physics surrogate: approximate the 1D heat equation solution
u(x, t) = sin(πx)·exp(-π²t) on a collocation grid.
"""
import torch
import photokan as pk


def main():
    torch.manual_seed(0)

    # Collocation points: (x, t) → u
    x = torch.rand(2000, 1) * 1.0
    t = torch.rand(2000, 1) * 0.5
    u = torch.sin(torch.pi * x) * torch.exp(-torch.pi**2 * t)

    inputs  = torch.cat([x, t], dim=1)     # [2000, 2]
    outputs = u                             # [2000, 1]

    x_tr, u_tr = inputs[:1600], outputs[:1600]
    x_te, u_te = inputs[1600:], outputs[1600:]

    model = pk.PhotoKAN(
        layer_sizes=[2, 16, 16, 1],
        activation="fourier",
        backend="auto",
        noise_sim=False,
        n_basis=8,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    for epoch in range(600):
        model.train()
        pred = model(x_tr)
        loss = torch.nn.functional.mse_loss(pred, u_tr)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                l2 = torch.sqrt(torch.nn.functional.mse_loss(model(x_te), u_te))
            print(f"Epoch {epoch:4d} | L2 error: {l2:.4f}")

    # Symbolic extraction
    print("\nSample symbolic formulas:")
    formulas = model.symbolic_regression()
    for k, expr in list(formulas.items())[:3]:
        print(f"  {k}: {expr}")


if __name__ == "__main__":
    main()
