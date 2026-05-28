"""
examples/mnist_photokan.py

PhotoKAN MNIST classification.
Target: ≥98% test accuracy (matching published photonic KAN benchmarks).

Run:
    python examples/mnist_photokan.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import photokan as pk

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.flatten()),   # 784-d flat vector
])

train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_data  = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE)

# ---------------------------------------------------------------------------
# Model: small KAN with standard linear input reduction
# ---------------------------------------------------------------------------
class PhotoKANMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        # Project 784 → 64 with a standard linear layer first
        # (full 784-wide KAN would be very slow without NPU)
        self.proj = nn.Linear(784, 64)
        self.kan = pk.PhotoKAN(
            layer_sizes=[64, 32, 10],
            activation="sine",
            backend="auto",
            n_basis=8,
            noise_sim=False,
        )

    def forward(self, x):
        x = F.relu(self.proj(x))
        return self.kan(x)

model = PhotoKANMNIST().to(DEVICE)
print(f"Device: {DEVICE}")
print(f"KAN params: {model.kan.parameter_count()['total']:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS
)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def evaluate(loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += len(yb)
    return correct / total


for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        loss = F.cross_entropy(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    if epoch % 5 == 0 or epoch == 1:
        acc = evaluate(test_loader)
        print(f"Epoch {epoch:3d}: loss={total_loss/len(train_loader):.4f}  "
              f"test_acc={acc*100:.2f}%")

print(f"\nFinal test accuracy: {evaluate(test_loader)*100:.2f}%")
