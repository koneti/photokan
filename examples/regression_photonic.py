#!/usr/bin/env python3
"""
PhotoKAN regression example.
Fits y = sin(x1) + x2^2 using a photonic KAN on CPU simulation.
Swap backend='qpal' when running on Q.ANT hardware.
"""

import torch
import photokan as pk

# ── Data ────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
x = torch.rand(1000, 2) * 4 - 2            # uniform in [-2, 2]
y = torch.sin(x[:, 0]) + x[:, 1] ** 2

x_train, x_test = x[:800], x[800:]
y_train, y_test = y[:800], y[800:]

# ── Model ────────────────────────────────────────────────────────────────────
model = pk.PhotoKAN(
    layer_sizes=[2, 8, 8, 1],
    activation="sine",
    backend="auto",         # ← use 'qpal' on NPU hardware
    n_basis=8,
    noise_sim=True,
)

print("Backend availability:", pk.available_backends())
print("Parameters:          ", model.parameter_count())
print("Op estimates vs MLP: ", model.estimate_ops())

# ── Training ─────────────────────────────────────────────────────────────────
optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

for epoch in range(500):
    model.train()
    pred = model(x_train).squeeze()
    loss = torch.nn.functional.mse_loss(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = model(x_test).squeeze()
            test_loss = torch.nn.functional.mse_loss(test_pred, y_test)
        print(f"Epoch {epoch:4d} | train {loss:.4f} | test {test_loss:.4f}")

# ── Symbolic regression ───────────────────────────────────────────────────────
print("\nSymbolic regression (sample edges):")
formulas = model.symbolic_regression()
for k, expr in list(formulas.items())[:4]:
    print(f"  {k}: {expr}")

# ── Compile (stub) ────────────────────────────────────────────────────────────
compiler = pk.PhotonicCompiler()
program  = compiler.compile(model, "/tmp/regression_model.npu")
print("\nCompiled bundle metadata:", program.metadata)
