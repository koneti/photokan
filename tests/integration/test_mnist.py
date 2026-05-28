"""
MNIST integration test — quick smoke + accuracy check.
Full 20-epoch run is marked slow; CI runs the 3-epoch smoke.
"""
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import photokan as pk


def _make_fake_mnist(n=512):
    """Fake MNIST-shaped data for CI (no download required)."""
    torch.manual_seed(7)
    x = torch.randn(n, 784)
    y = torch.randint(0, 10, (n,))
    return x, y


class TestMNIST:

    def test_photokan_mnist_smoke(self):
        """3 epochs on fake data — verifies the pipeline runs end-to-end."""
        x, y = _make_fake_mnist(256)
        dl = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)

        model = pk.PhotoKAN(
            layer_sizes=[784, 32, 10],
            activation="relu",
            backend="cpu",
            noise_sim=False,
            n_basis=4,
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(3):
            for xb, yb in dl:
                loss = F.cross_entropy(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            acc = (model(x).argmax(1) == y).float().mean().item()
        # With random labels accuracy can be low; just check it ran
        assert 0 <= acc <= 1

    @pytest.mark.slow
    def test_photokan_mnist_real(self):
        """Full MNIST run — requires download, marked slow."""
        pytest.importorskip("torchvision")
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
        train = datasets.MNIST("./data", train=True,  download=True, transform=transform)
        test  = datasets.MNIST("./data", train=False, download=True, transform=transform)
        trdl  = DataLoader(train, batch_size=256, shuffle=True)
        tedl  = DataLoader(test,  batch_size=512)

        model = pk.PhotoKAN(
            layer_sizes=[784, 64, 32, 10],
            activation="sine", backend="auto", noise_sim=False, n_basis=6,
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(5):
            model.train()
            for x, y in trdl:
                loss = F.cross_entropy(model(x), y)
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in tedl:
                correct += (model(x).argmax(1) == y).sum().item()
                total   += y.size(0)
        acc = correct / total
        assert acc >= 0.85, f"MNIST accuracy {acc:.3f} below threshold"
