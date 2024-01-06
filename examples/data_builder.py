import datasets
from oobleck_colossalai import HeterogeneousDataLoader, HeterogeneousParallelPlugin
from transformers import AutoTokenizer, PreTrainedTokenizer


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

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
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
        pad_tokens: bool = True,
        task_name: str = "mrpc",
        max_seq_length: int = 128,
    ):
        self.plugin = plugin
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True
        )
        if pad_tokens:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
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

    def dataloader(self) -> HeterogeneousDataLoader:
        return self.plugin.prepare_dataloader(
            self.dataset["train"],
            shuffle=True,
            drop_last=True,
        )
