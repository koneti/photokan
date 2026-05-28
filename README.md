# PhotoKAN

**Photonic Kolmogorov-Arnold Networks** — vendor-agnostic PyTorch framework for photonic AI hardware.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1%2B-orange)](https://pytorch.org)
[![PyPI](https://img.shields.io/pypi/v/photokan)](https://pypi.org/project/photokan/)

---

## Why PhotoKAN?

Standard MLPs use fixed activations on *nodes* and linear weights on *edges*.
KANs invert this: **learnable nonlinear functions sit on the edges**, summed at nodes.

Photonic hardware is physically structured around edge-level nonlinear transforms —
light through a waveguide produces exactly the kind of parametric nonlinear function
a KAN edge needs. This is not an analogy; it is a direct structural match.

Published benchmarks show:
- **43% fewer parameters** vs equivalent MLPs
- **46% fewer operations** vs equivalent MLPs
- **30x energy efficiency** gain on photonic NPU vs CMOS

---

## Supported Hardware

| Vendor | Technology | SDK | Install |
|--------|-----------|-----|---------|
| **Q.ANT** | Thin-Film Lithium Niobate (TFLN) | Q.PAL | `pip install photokan[qant]` |
| **Lightmatter** | Silicon Photonics | Lightmatter SDK | `pip install photokan[lightmatter]` |
| **Salience Labs** | III-V Photonics (InP) | Salience SDK | `pip install photokan[salience]` |
| **CPU Sim** | Physics-accurate noise model | Built-in | `pip install photokan` |

---

## Installation

```bash
pip install photokan                        # CPU simulation (no hardware required)
pip install photokan[qant]                  # + Q.ANT NPU support
pip install photokan[lightmatter]            # + Lightmatter support
pip install photokan[salience]              # + Salience Labs support
pip install photokan[llm]                   # + HuggingFace / PEFT integration
pip install photokan[onnx]                  # + ONNX export
pip install photokan[dev]                   # + development tools
```

---

## Quick Start

```python
import torch
import photokan as pk

# Works on CPU sim if no hardware present — no code changes needed
model = pk.PhotoKAN(
    layer_sizes=[4, 16, 16, 1],
    activation='sine',   # 'sine' | 'fourier' | 'spline' | 'relu'
    backend='auto',      # auto-detects photonic hardware, falls back to CPU
)

x = torch.randn(32, 4)
y = model(x)

# Standard PyTorch training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = torch.nn.MSELoss()(y, torch.randn(32, 1))
loss.backward()
optimizer.step()

# Check what hardware is available
print(pk.available_backends())
# -> {'cpu': True, 'cuda': False, 'qant': False, 'lightmatter': False, 'salience': False}

# List registered vendors
print(pk.all_vendor_names())
# -> ['qant', 'lightmatter', 'salience']

# Get vendor-specific noise profiles
print(pk.get_noise_config('qant', 'npu1'))
# -> {'snr_db': 14.0, 'bit_depth': 6, 'phase_noise_rad': 0.02, 'technology': 'tfln', 'enabled': True}
```

---

## Vendor-Specific Dispatch

Target a specific vendor or let the framework auto-detect:

```python
model = pk.PhotoKAN(
    layer_sizes=[4, 16, 1],
    activation='sine',
    backend='qant',        # explicitly use Q.ANT hardware
)

# Or simulate a specific vendor's noise characteristics on CPU
model = pk.PhotoKAN(
    layer_sizes=[4, 16, 1],
    activation='sine',
    backend='cpu',
    noise_config=pk.get_noise_config('lightmatter', 'envise1'),
)
```

---

## Activation Variants

| Name | Formula | Best for | Photonic native |
|------|---------|----------|-----------------|
| `sine` | `sum w*sin(f*x + p)` | Periodic targets, photonic deployment | Yes |
| `fourier` | `a0 + sum [a*cos + b*sin]` | Multi-frequency signals | Yes |
| `spline` | B-spline basis | Non-periodic, high precision | Via LUT |
| `relu` | `sum w*ReLU(a*x + b)` | Edge inference, speed | Yes |

---

## Photonic Noise Simulation

Test accuracy against realistic hardware impairments before deploying:

```python
sim = pk.PhotonicSimulator()

# Use vendor-specific noise profile
sim.set_hardware_profile('npu2')   # Q.ANT NPU2: SNR=16dB, 8-bit

results = sim.sweep_snr(model, x_test, y_test,
                         snr_range=[8, 10, 12, 14, 16, 20])
sim.plot_snr_accuracy(results)
```

### Noise Profiles by Vendor

**Q.ANT (TFLN)**
| Profile | SNR | Bit Depth | Phase Noise |
|---------|-----|-----------|-------------|
| `npu1` | 14 dB | 6-bit | 0.02 rad |
| `npu2` | 16 dB | 8-bit | 0.01 rad |
| `ideal` | 60 dB | 16-bit | 0.0 rad |

**Lightmatter (Silicon Photonics)**
| Profile | SNR | Bit Depth | Thermal Drift |
|---------|-----|-----------|---------------|
| `envise1` | 12 dB | 5-bit | 0.005 |
| `mars1` | 15 dB | 7-bit | 0.002 |
| `ideal` | 60 dB | 16-bit | 0.0 |

**Salience Labs (III-V/InP)**
| Profile | SNR | Bit Depth | Ring Crosstalk |
|---------|-----|-----------|----------------|
| `mr100` | 18 dB | 8-bit | 0.003 |
| `mr200` | 22 dB | 10-bit | 0.001 |
| `ideal` | 60 dB | 16-bit | 0.0 |

---

## Energy Estimation

Estimate energy consumption with published photonic efficiency numbers:

```python
from photokan.utils import estimate_model_energy

reports = estimate_model_energy(model, batch_size=64)
# Layer 0: 8.214 uJ (CMOS) -> 0.267 uJ (Photonic), 30.8x better
# Layer 1: ...
```

---

## Convolutional KAN

Use `PhotoConvKAN` for image and spatial workloads:

```python
model = pk.PhotoConvKAN(
    in_channels=1, out_channels=16,
    kernel_size=3, activation='sine',
)
x = torch.randn(8, 1, 28, 28)
y = model(x)  # [8, 16, 28, 28]
```

---

## LLM Integration

Replace MLP layers in HuggingFace transformers with PhotoKAN using LoRA-style adapters:

```python
from photokan.llm import add_photo_lora, PhotoKANAttention

# Wrap a transformer's MLP layers
model = add_photo_lora(model, target_modules=["mlp"], n_basis=4)
```

---

## AOT Compilation

Compile models to photonic deployment bundles (`.npu`):

```python
compiler = pk.PhotonicCompiler()
program = compiler.compile(model, './my_model.npu')

# CPU validation (LUT interpreter)
y = program.run(x, backend='cpu')

# Hardware inference
y = program.run(x, backend='qant')      # Q.ANT
y = program.run(x, backend='lightmatter')  # Lightmatter

# Benchmark latency
stats = program.benchmark(x)
# -> {'mean_ms': 0.42, 'throughput_samples_per_sec': 76190}
```

---

## Architecture

```
User PyTorch model
  |- PhotoKAN / PhotoKANLayer / PhotoConvKAN  (nn.Module, drop-in)
       |- EdgeActivations (sine / fourier / spline / relu)
            |- SimBackend -> CPU physics simulation (vendor noise profiles)
            |- Vendor dispatch -> Q.ANT / Lightmatter / Salience via pluggable backends
  |- PhotonicCompiler
       |- LUTCompiler -> int8 quantised lookup tables
       |- GraphBuilder -> photonic execution graph
       |- PhotonicProgram -> run on hardware or CPU LUT interpreter
  |- PhotonicSimulator -> SNR sweeps, transfer functions
  |- LLM integration (LoRA adapters, attention)
```

---

## Adding a New Vendor

PhotoKAN uses a pluggable backend architecture. To add support for a new photonic vendor:

1. Create `photokan/backends/<vendor>/__init__.py` and `backend.py`
2. Implement the `PhotonicBackend` ABC:

```python
from photokan.backends.base import PhotonicBackend
from photokan.backends.registry import register_vendor

class MyBackend(PhotonicBackend):
    @staticmethod
    def name() -> str: ...
    @staticmethod
    def is_available() -> bool: ...
    @staticmethod
    def execute(x, activation, op_type) -> torch.Tensor: ...
    @staticmethod
    def compute_gradient(grad_output, x, activation, op_type) -> tuple: ...
    @staticmethod
    def noise_profiles() -> dict[str, dict]: ...

register_vendor(MyBackend)
```

3. Add the vendor to `_discover_vendors()` in `registry.py`

---

## Project Status

| Phase | Features | Status |
|-------|----------|--------|
| **Phase 1** | Activations, SimBackend, Layers, Energy, Profiler | Done |
| **Phase 2** | LUT compiler, execution graph, ONNX export, ConvKAN | Done |
| **Phase 3** | Vendor-agnostic backends (Q.ANT, Lightmatter, Salience) | Done |
| **Phase 4** | Hardware dispatch, LLM fine-tuning, arXiv paper | Upcoming |

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest                          # 136+ tests
pytest -k "not slow"            # skip slow tests
pytest --cov=photokan           # with coverage
```

---

## References

- Liu et al. (2024) -- KAN: Kolmogorov-Arnold Networks (arXiv 2404.19756)
- Peng et al. (2024) -- Photonic KAN via RAMZI (98% MNIST, 65x energy-area reduction)
- Reinhardt et al. (2024) -- SineKAN
- Q.ANT NPU -- https://qant.com/photonic-computing/
- Lightmatter -- https://lightmatter.co
- Salience Labs -- https://salience.io

---

*PhotoKAN -- Build the bridge. Light does the math.*
