# Cornstarch examples

These examples use synthetic data and random model initialization so the code
stays focused on Cornstarch's model construction, execution-plan, and
parallelization APIs. Model configs, tokenizers, and modality processors are
loaded from Hugging Face.

Run the local examples from the repository root on a CUDA GPU:

```bash
python -m examples.pretrain_vlm
python -m examples.pretrain_valm
```

The distributed examples model a production one-process-per-GPU setup. They
require CUDA and NCCL, bind each process to `LOCAL_RANK` with
`torch.cuda.set_device()`, and support single- or multi-node `torchrun`
launches. For example:

```bash
# Eight GPUs: DP=2, PP=2, TP=2.
torchrun --nproc-per-node=8 --module examples.distributed.pretrain_llm \
    --dp 2 --pp 2 --tp 2

# Three GPUs: one vision pipeline stage plus a two-way-TP language stage.
torchrun --nproc-per-node=3 --module examples.distributed.pretrain_vlm \
    --llm-pp 1 --llm-tp 2
```

For a multi-node launch, add torchrun's `--nnodes`, `--node-rank`,
`--master-addr`, and `--master-port` arguments. The product of each module's
parallel dimensions and its data-parallel size must match the ranks assigned
to that module. In the VLM example, the vision encoder is deliberately not
tensor-parallelized because Cornstarch currently registers tensor-parallel
plans for the supported language-model families, not CLIP.

`benchmark_gated_delta_cp` and `validate_qwen_full_parallel` are specialized
release/benchmark programs. Their module docstrings list topology and optional
Flash Linear Attention requirements.
