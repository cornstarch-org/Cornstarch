import torch.nn as nn

from .auto_module import BaseModuleInfo


class GPT2Model(BaseModuleInfo):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        if self.model.__class__.__name__ == "GPT2Model":
            module = self.model
        else:
            module = self.model.transformer

        return (
            ["wte", "wpe", "drop"] + [f"h.{i}" for i in range(len(module.h))] + ["ln_f"]
        )


class GPT2LMHeadModel(GPT2Model):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        return [f"transformer.{module}" for module in super().modules()] + ["lm_head"]


class GPT2DoubleHeadsModel(GPT2Model):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        return [f"transformer.{module}" for module in super().modules()] + [
            "lm_head",
            "multiple_choice_head",
        ]


class GPT2ForQuestionAnswering(GPT2Model):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        return [f"transformer.{module}" for module in super().modules()] + [
            "qa_outputs"
        ]


class GPT2ForTokenClassification(GPT2Model):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        return [f"transformer.{module}" for module in super().modules()] + [
            "dropout",
            "classifier",
        ]


class GPT2ForSequenceClassification(GPT2Model):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        return [f"transformer.{module}" for module in super().modules()] + ["score"]
