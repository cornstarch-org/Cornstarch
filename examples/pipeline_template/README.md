# How to run

1. Modify template configuration as you want.
2. Use torchrun to run distributed training.


## Modifying template configuration
In each script, you will see the following code:

```python
template = PipelineTemplate(model_name, [modules_per_stage])
plugin = HeterogeneousParallelPlugin(
    tp_size=...,
    ...
)
plugin.set_pipelines(pipelines=[list of templates...], num_microbatches={dict template: integer})
```

First, define template(s) as you want. The second argument of pipeline template creation is a list of modules **per stage (type: list[list[str]])**.
You can get the entire list of modules via `PipelineTemplate.get_modules(model)`, where `model` is a HuggingFace model that this backend supports, and split this list to form a list of modules per stage.

> If the given model is not supported, it will raise `NotImplementedError: Auto policy for {model_name} is not implemented...`.

Examples of the list of modules per stage, if the whole list of modules is stored in `modules`, look like:

- 1-stage pipeline template: `[modules]`
- 2-stage pipeline template: `[modules[:4], modules[4:]]` (the first stage will have the first 4 modules, and the second stage will take all the other; feel free to change the number of layers)
- 3-stage pipeline template: `[modules[:6], modules[6:12], modules[12:]]`
...

> All modules returned from `PipelineTemplate.get_modules(model)` must be covered, otherwise it will raise `ValueError: Modules in the pipeline template do not match the modules in the model`.

> Some modules must be in specific stage, e.g. `transformer.embeddings.wte`, `transformer.embeddings.wpe` in `GPT2LMHeadModel` should be in the first stage, otherwises it will raise `ValueError: The first (or last) stage must contain ... module`.

Next, call `set_pipelines` to manually instantiate pipeline templates.
For example, if you want to instantiate two 1-stage pipelines, the list of pipeline templates should include two 1-sage pipeline templates: `pipelines=[template_1stage, template_1stage]`.
You may mix heterogeneous pipeline templates: `pipelines=[template_1stage, template_2stages]`.

> You should match torch.distributed world size and total number of ranks used in heterogeneous pipeline parallel training. Currently, the number of processes it requires is: `sum(number of stages per pipeline) * tp_size`.
Otherwise, it will raise `AssertionError: Number of ranks in pipeline templates does not match world size`.

Third, properly set the number of microbatches per pipeline template. Homogeneous pipelines will have the same number of microbatches (i.e. if you instantiate `template_1stage` twice, they cannot have different number of microbatches), thus the seconcd argument of `plugin.set_pipelines` is a dictionary of `PipelineTemplate -> number of microbatches`.

> All pipeline templates given in the first argument `pipelines` should also be included in keys of `num_microbatches`, otherwise it will raise `AssertionERror: All pipelines must have a corresponding number of microbatches`.

## Let's run!

```bash
torchrun (--nnodes N | --standalone) --nproc-per-node P run_<model>.py <script_specific_arguments...>
```

> You may use another multi-node runner as long as it properly configures required OS environments for `torch.distributed` initialization: `RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT`.

Example:
```baseh
torchrun --standalone --nproc-per-node 4 run_gpt2.py
```
It uses 4 GPUs in the local machine (no remote node involved) and runs GPT2 training.

```bash
torchrun --nnodes 2 --nproc-per-node 4 --node_rank [0|1] --master_addr MASTER_ADDR --master_port 29500 run_gpt2.py
```
It uses total 8 GPUs (2 nodes, 4 GPUs each) to run GPT2 training. Both nodes should have at least 4 GPUs. The command should be run on both node manually, with different `node_rank` (run on rank 0 first).