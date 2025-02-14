# Distributed Training Overview

Multimodal LLMs made by Cornstarch can be parallelized using either PyTorch Data Parallel (DDP or FSDP), ColossalAI plugins (for  tensor parallelism only), or Cornstarch plugins (for 4D parallelism).

- [Using DDP/FSDP](ddp_fsdp.md)
- [Cornstarch 4D Parallelism](cornstarch_parallel.md)