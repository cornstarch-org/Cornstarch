from .auto_module import BaseModuleInfo

import torch.nn as nn


class BloomModel(BaseModuleInfo):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        if self.model.__class__.__name__ == "BloomModel":
            module = self.model
        else:
            module = self.model.transformer

        return (
            ["word_embeddings", "word_embeddings_layernorm"]
            + [f"h.{i}" for i in range(len(module.h))]
            + ["ln_f"]
        )


class BloomForCausalLM(BloomModel):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        return [f"transformer.{module}" for module in super().modules()] + ["lm_head"]


class BloomForSequenceClassification(BloomModel):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        return [f"transformer.{module}" for module in super().modules()] + ["score"]


class BloomForTokenClassification(BloomModel):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        return [f"transformer.{module}" for module in super().modules()] + [
            "dropout",
            "classifier",
        ]


class BloomForQuestionAnswering(BloomModel):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        return [f"transformer.{module}" for module in super().modules()] + [
            "qa_outputs",
        ]
