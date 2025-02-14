<h1 align="center">Cornstarch<br>
A Versatile Model Toolkit for Distributed Training</h1>

Cornstarch includes tools that change HuggingFace model shapes for various distributed training.
Powered by [ColossalAI](https://colossalai.org/) for basic parallelism implementation and training backend, Cornstarch provides more various ways of changing the form of distributed training.

## Install

Due to conflict on version dependency between ColossalAI and Cornstarch, we recommend to follow the instruction below step by step.

1. First, install colossalai:
```
pip install colossalai
```
2. Then, install cornstarch:
```
pip install --ignore-installed cornstarch
```
This will upgrade `transformers` and 

Then, upgrade `transformers` to 4.40 to use more recent models, e.g. phi-3 or gemma.
```
pip install -U transformers==4.40.*
```
You will see an error from pip's dependency resolve about version dependency conflict, Cornstarch will address the compatibility issue.

Optionally, install [`apex`](https://github.com/nvidia/apex) to use fused normalization feature.

## Run

See [how to run](examples/README.md).

## Debug tests without torch.compile
ColossalAI uses `@torch.compile`, which spawns async compile worker processes (32 workers by default) **per process** you spawn.
It drastically slows down debugging test cases especially when a test case runs with `torch.multiprocessing` (e.g. tests in `tests/test_shardformer`).

Disable torch.compile with the following environment variable:
```bash
export TORCHDYNAMO_DISABLE=1
```

Or use `pytest-env` package. Install the package via `pip install pytest-env==1.1.3`.
Cornstarch `pyproject.toml` includes a configuration for `pytest-env` (`[tool.pytest.ini_options]`) and automatically injects the environment variable above to your tests.