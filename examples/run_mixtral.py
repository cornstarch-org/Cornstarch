import functools

import click
import colossalai
import datasets
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.nn.optimizer import FusedAdam
from tqdm import tqdm
from transformers import PreTrainedTokenizer, get_linear_schedule_with_warmup
from transformers.models.llama import LlamaTokenizerFast
from transformers.models.mixtral import MixtralConfig, MixtralForCausalLM

from cornstarch.shardformer.policies.mixtral import MixtralForCausalLMPolicy


def tokenize_batch_for_pretrain(
    batch, tokenizer: PreTrainedTokenizer | None = None, max_length: int = 2048
):
    texts = [sample["text"] for sample in batch]
    data = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    data = {k: v.cuda() for k, v in data.items()}
    data["labels"] = data["input_ids"].clone()
    return data


@click.command
@click.option(
    "--model_name_or_path",
    type=str,
    default="mistralai/Mixtral-8x7B-Instruct-v0.1",
    help="Model name or path.",
)
@click.option("--global_batch_size", type=int, default=1, help="Global batch size.")
def main(
    model_name_or_path: str,
    global_batch_size: int,
):
    colossalai.launch_from_torch(config={})
    coordinator = DistCoordinator()

    with LazyInitContext(default_device=get_accelerator().get_current_device()):
        config = MixtralConfig.from_pretrained(model_name_or_path)
        # TODO: remove it when example starts using pipeline parallelism
        config.num_hidden_layers = 6
        config._attn_implementation = "flash_attention_2"
        model = MixtralForCausalLM.from_pretrained(model_name_or_path, config=config)
    model.gradient_checkpointing_enable()

    policy = MixtralForCausalLMPolicy()
    policy.set_model(model)

    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")["train"]
    tokenizer = LlamaTokenizerFast.from_pretrained(
        "mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    tokenizer.pad_token = tokenizer.eos_token

    plugin = HybridParallelPlugin(
        tp_size=4, pp_size=1, precision="bf16", custom_policy=policy
    )
    dataloader = plugin.prepare_dataloader(
        dataset,
        shuffle=True,
        drop_last=True,
        batch_size=global_batch_size,
        collate_fn=functools.partial(tokenize_batch_for_pretrain, tokenizer=tokenizer),
    )

    booster = Booster(plugin=plugin)
    total_steps = len(dataloader)
    optimizer = FusedAdam(model.parameters())
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=total_steps,
    )

    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model, optimizer, dataloader=dataloader, lr_scheduler=lr_scheduler
    )

    # Train model
    model.train()
    optimizer.zero_grad()

    dataloader_iter = iter(dataloader)
    with tqdm(range(total_steps), disable=not (coordinator.is_master())) as pbar:
        for _ in pbar:
            inputs = next(dataloader_iter)
            outputs = model(**inputs)
            loss = outputs.loss
            optimizer.backward(loss)

            pbar.set_postfix({"loss": loss.item()})

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()


if __name__ == "__main__":
    main()
