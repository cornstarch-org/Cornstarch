# Installing Cornstarch

Cornstarch is a Python library that works with Pytorch (>=2.5) and ColossalAI (>=0.4.4).

## Requirments

- OS: Linux.
- Python 3.10 and above.
- Pytorch 2.5 and above.
- Ampere or Hopper GPUs. Does not guarantee working on other generations of NVIDIA GPUs or AMD GPUs.

!!! note

    It is highly recommended to use [a Pytorch Docker container](https://hub.docker.com/r/pytorch/pytorch) that provides pre-compiled CUDA, NCCL, Python, and Pytorch, and then install `colossalai` and `Cornstarch` in it.
    For systems powered by ARM CPUs (e.g. Altera or Grace), use an [NVIDIA NGC Pytorch Docker container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

    Follow [NVIDIA Container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit) guideline to use GPUs inside a Docker container.

## Install Cornstarch from PyPI

Cornstarch relies on [Colossal-AI](https://github.com/hpcaitech/ColossalAI) for training a multimodal LLM, which has a strict version requirement on [Pytorch](https://github.com/pytorch/pytorch) and [Huggingface transformers](https://github.com/huggingface/transformers) that are not compatible with Cornstarch dependencies.

For this reason, ColossalAI should first manually be installed and then should Cornstarch be installed.

```
$ pip install colossalai
$ pip install cornstarch
```

During installing `colossalai`, your preinstalled `torch` and `transformers` will be uninstalled.  
To avoid redundant installation, you can use `--no-deps` when installing `colossalai`:

```
$ pip install --no-deps colossalai
$ pip install cornstarch
```

Cornstarch includes every dependent package that `colossalai` has, thus it should have no problem in using `colossalai` or Cornstarch.

## Install Cornstarch from Source

You can also install Cornstarch from source by cloning the Github repository. Manual `colossalai` installation introduced above should still be done.

```
$ pip install --no-deps colossalai
$ git clone https://github.com/SymbioticLab/Cornstarch
$ cd Cornstarch
$ pip install -e .[dev]
```