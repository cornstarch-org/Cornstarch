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

For this reason, ColossalAI should first manually be installed and then should Cornstarch be installed:

```
$ pip install --no-deps colossalai==0.4.6
$ pip install cornstarch
```

!!! note

    You will see an error message from pip dependency resolver similar to the following:

    ```
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    colossalai 0.4.6 requires torch<=2.4.1,>=2.2.0, but you have torch <your_version> which is incompatible.
    colossalai 0.4.6 requires transformers==4.39.3, but you have transformers <your_version> which is incompatible.
    ```

    But this is expected and there is no problem in installing and using cornstarch.
    You can also see that packages including cornstarch are successfully installed right after the error message:
    ```
    Successfully installed ... cornstarch-0.0.5 ... tokenizers-0.21.0 transformers-4.49.0 ...
    ```

## Install Cornstarch from Source

You can also install Cornstarch from source by cloning the Github repository. Manual `colossalai` installation introduced above should still be done.

```
$ pip install --no-deps colossalai==0.4.6
$ git clone https://github.com/cornstarch-org/Cornstarch
$ cd Cornstarch
$ pip install -e .[dev]
```