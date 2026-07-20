---
description: "Cornstarch: Multimodal Model Training Framework"
hide:
  - toc
  - navigation
---
<div align="center">
<img src="assets/images/cornstarch.svg">
<h1><strong>Cornstarch</strong></h1>
<h2>Build, Train, and Run Your Own Multimodal Model</h2>
</div>
---

Cornstarch converts supported Hugging Face components into unified, lazily
initialized modules; connects them through a user-defined execution DAG; and
applies a distinct composable parallelism plan to each module.

# Documentation Organization
- [Core architecture](architecture.md): The five invariants behind models, DAGs,
  lazy initialization, and per-module parallelism.
- [Getting Started](getting_started/installation.md): Instructions on installation and setup.
- [Creating a multimodal LLM](using_cornstarch/creating_mllm.md): How to create a multimodal LLM from unimodal models.
- [Preprocessing multimodal inputs](using_cornstarch/preprocessing_inputs.md): How to preprocess muiltimodal inputs to run a multimodal LLM.
- [Training a multimodal LLM](using_cornstarch/training_mllm.md): How to train a multimodal LLM using Cornstarch.
- [Multimodal LLM Parallelization](parallelization/index.md): How to parallelize a multimodal LLM training.

# Research Works
Below is a list of works powered by Cornstarch.

- [Oobleck](https://github.com/SymbioticLab/Oobleck): resilient distributed training using pipeline template

## Contact
Insu Jang (insujang@umich.edu)
