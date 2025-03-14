<div align="center">

![Cornstarch Logo](https://cornstarch-org.github.io/assets/images/cornstarch.svg)

<h1>Cornstarch<br>
Build, Train, Run Your Own Multimodal Model</h1>
</div>

Cornstarch is a multimodal model training framework, including distributed training features with 5D parallelism (PP, TP, CP, DP, and modality parallelism).
You can create your own multimodal model with a set of HuggingFace unimodal models and train it.

Cornstarch provides

- **Pipeline Template and Heterogeneous Pipeline Parallelism**: specify different pipeline templates and combine them to deploy heterogeneous pipeline parallel execution
- **Composable multimodal model creation**: specify your own multimodal models from a set of HuggingFace transformers unimodal models
- **MultimodalModel Generation and Parallelization**: specify your own multimodal model and parallelize it with 5D parallelism (DP+PP+TP+CP+and modality parallelism)

## Install and Run

Please refer to [our document](https://cornstarch-org.github.io/getting_started/installation/)!

## Research

A technial report will be released very soon.

## Contact

- Insu Jang (insujang@umich.edu)