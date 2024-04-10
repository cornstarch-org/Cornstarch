<h1 align="center">Cornstarch<br>
A Versatile Model Toolkit for Distributed Training</h1>

Cornstarch includes tools that change HuggingFace model shapes for various distributed training.
Powered by [ColossalAI](https://colossalai.org/) for basic parallelism implementation and training backend, Cornstarch provides more various ways of changing the form of distributed training.

## Install

Use `pip` to install Cornstarch:
```
pip install cornstarch
```

Optionally, install [`apex`](https://github.com/nvidia/apex), [`xformers`](https://github.com/facebookresearch/xformers) and [`flash-attn`](https://github.com/Dao-AILab/flash-attention) to boost throughput (follow instructions in each README).

## Run

See [how to run](examples/README.md).