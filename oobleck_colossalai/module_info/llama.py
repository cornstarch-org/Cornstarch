from .auto_module import BaseModuleInfo

import torch.nn as nn


class LlamaModel(BaseModuleInfo):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        if self.model.__class__.__name__ == "LlamaModel":
            module = self.model
        else:
            module = self.model.model

        return (
            ["embed_tokens"]
            + [f"layers.{i}" for i in range(len(module.layers))]
            + ["norm"]
        )


class LlamaForCausalLM(LlamaModel):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        return [f"model.{module}" for module in super().modules()] + ["lm_head"]


class LlamaForSequenceClassification(LlamaModel):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        return [f"model.{module}" for module in super().modules()] + ["score"]
