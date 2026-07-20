# Distributed training

Cornstarch's native `ParallelizationPlan` composes data, pipeline, context,
tensor, and expert parallelism independently for each module in a multimodal
DAG. The plan is the supported distributed interface; legacy ColossalAI plugin
and fixed `MultimodalModel` documentation does not describe the active package.

- [Core architecture](../architecture.md)
- [Using Cornstarch parallelism](cornstarch_parallel.md)
- [PyTorch DDP/FSDP notes](ddp_fsdp.md)
