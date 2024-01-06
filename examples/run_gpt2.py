from dataclasses import dataclass

import colossalai
import datasets
import simple_parsing
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    GPT2ForSequenceClassification,
    PretrainedConfig,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from oobleck_colossalai import (
    HeterogeneousDataLoader,
    HeterogeneousParallelPlugin,
    PipelineTemplate,
)
from oobleck_colossalai.module_info.auto_module import get_module_names


class GLUEDataBuilder:
    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        plugin: HeterogeneousParallelPlugin,
        task_name: str = "mrpc",
        max_seq_length: int = 128,
    ):
        self.plugin = plugin
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset = datasets.load_dataset("glue", task_name)

        def convert_to_features(example_batch):
            text_fields = GLUEDataBuilder.task_text_field_map[task_name]
            if len(text_fields) > 1:
                texts_or_text_pairs = list(
                    zip(example_batch[text_fields[0]], example_batch[text_fields[1]])
                )
            else:
                texts_or_text_pairs = example_batch[text_fields[0]]

            # Tokenize the text/text pairs
            features = self.tokenizer.batch_encode_plus(
                texts_or_text_pairs,
                max_length=max_seq_length,
                padding="max_length",
                truncation=True,
            )

            features["labels"] = example_batch["label"]
            return features

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [
                c
                for c in self.dataset[split].column_names
                if c in GLUEDataBuilder.loader_columns
            ]
            self.dataset[split].set_format(type="torch", columns=self.columns)

    def dataloader(self) -> DataLoader:
        return self.plugin.prepare_dataloader(
            self.dataset["train"],
            shuffle=True,
            drop_last=True,
        )


@dataclass
class ExampleArguments:
    model_name_or_path: str = "gpt2"
    global_batch_size: int = 32
    num_epoch: int = 3
    warmup_faction: float = 0.1


def main():
    args: ExampleArguments = simple_parsing.parse(ExampleArguments)

    colossalai.launch_from_torch(config={})
    coordinator = DistCoordinator()

    plugin = HeterogeneousParallelPlugin(
        tp_size=1,
        global_batch_size=args.global_batch_size,
        microbatch_size=4,
        precision="bf16",
    )
    booster = Booster(plugin=plugin)

    config: PretrainedConfig = AutoConfig.from_pretrained(args.model_name_or_path)
    config.pad_token_id = config.eos_token_id
    model = GPT2ForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config
    )

    modules = get_module_names(model)
    template1 = PipelineTemplate(modules_per_stage=[modules])
    template2 = PipelineTemplate(modules_per_stage=[modules[:8], modules[8:]])
    plugin.set_pipeline_templates(
        # homogeneous pipelines with 4 GPUs
        # pipeline_templates={template2: 2},
        # num_microbatches={template2: 4},
        # heterogeneous pipelines with 3 GPUs
        pipeline_templates={template1: 1, template2: 1},
        num_microbatches={template1: 3, template2: 5},
    )

    # Prepare dataloader
    data_builder = GLUEDataBuilder(args.model_name_or_path, plugin, task_name="mrpc")
    dataloader: HeterogeneousDataLoader = data_builder.dataloader()

    # optimizer
    optimizer = Adam(model.parameters())

    # lr_scheduler
    total_steps = len(dataloader) * args.num_epoch
    num_warmup_steps = int(total_steps * args.warmup_faction)
    lr_scheduler: LambdaLR = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    model, optimizer, _criterion, _, lr_scheduler = booster.boost(
        model,
        optimizer=optimizer,
        criterion=lambda outputs, inputs: outputs.loss,
        dataloader=dataloader,
        lr_scheduler=lr_scheduler,
    )

    # Train model
    model.train()
    optimizer.zero_grad()
    is_pp_last_stage = plugin.stage_manager.is_last_stage()

    for epoch in range(args.num_epoch):
        total_step = len(dataloader)
        dataloader_iter = iter(dataloader)
        with tqdm(
            range(total_step),
            desc=f"Epoch [{epoch + 1}/{args.num_epoch}]",
            disable=not (coordinator.is_master() or is_pp_last_stage),
        ) as pbar:
            for _ in pbar:
                outputs = booster.execute_pipeline(
                    dataloader_iter,
                    model,
                    criterion=lambda outputs, inputs: outputs.loss,
                    optimizer=optimizer,
                    return_loss=True,
                    return_outputs=True,
                )

                if is_pp_last_stage:
                    loss = outputs["loss"]
                    pbar.set_postfix({"loss": loss.item()})

                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()


if __name__ == "__main__":
    main()
