# Installing Cornstarch

Cornstarch requires Linux, Python 3.10 or newer, and PyTorch 2.6 or newer. The
distributed runtime is implemented with native PyTorch DTensor, device meshes,
and process groups; ColossalAI is not part of the active architecture.

CUDA is required for the optimized training kernels. CPU plus gloo is useful for
the distributed correctness tests, but it is not the intended training setup.

## From PyPI

```bash
pip install cornstarch
```

For GPU installations, start from a PyTorch or NVIDIA PyTorch container whose
CUDA version matches the installed PyTorch build. This avoids rebuilding CUDA,
NCCL, and the compiler toolchain from scratch.

## From source

```bash
git clone https://github.com/cornstarch-org/Cornstarch
cd Cornstarch
pip install -e '.[dev]'
```

Install documentation dependencies with `pip install -e '.[docs]'` and build
the site with `mkdocs build --strict`.
