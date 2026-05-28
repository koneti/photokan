#!/usr/bin/env python3
"""
PhotoKAN MNIST classifier.
Target: ≥ 97% accuracy using a [784, 64, 32, 10] photonic KAN.
Run: python examples/mnist_photokan.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import photokan as pk


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1)),   # flatten to 784
    ])
    train_ds = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2)

    model = pk.PhotoKAN(
        layer_sizes=[784, 64, 32, 10],
        activation="sine",
        backend="auto",
        noise_sim=False,
        n_basis=6,
    ).to(device)

    print(f"Parameters: {model.parameter_count()['total']:,}")
    print(f"Ops vs MLP: {model.estimate_ops()}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    for epoch in range(1, 21):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

        if epoch % 5 == 0:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in test_dl:
                    x, y = x.to(device), y.to(device)
                    correct += (model(x).argmax(1) == y).sum().item()
                    total   += y.size(0)
            print(f"Epoch {epoch:2d} | test accuracy: {100*correct/total:.2f}%")

    # Compile to .npu
    compiler = pk.PhotonicCompiler(n_lut_points=256, max_lut_mse=1e-3)
    prog = compiler.compile(model.cpu(), "./mnist_model.npu", validate=False)
    print(f"\nCompiled bundle: {prog.inspect()}")


if __name__ == "__main__":
    main()
