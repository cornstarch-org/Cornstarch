from .auto_module import BaseModuleInfo

import torch.nn as nn


class BertModel(BaseModuleInfo):
    def __init__(self, model: nn.Module, add_pooling_layer: bool = True):
        super().__init__(model)
        self.add_pooling_layer = add_pooling_layer

    def modules(self) -> list[str]:
        if self.model.__class__.__name__ == "BertModel":
            module = self.model
        else:
            module = self.model.bert

        modules = ["embeddings"] + [
            f"encoder.layer.{i}" for i in range(len(module.encoder.layer))
        ]
        if self.add_pooling_layer:
            modules += ["pooler"]

        return modules


class BertForPreTraining(BertModel):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        return [f"bert.{module}" for module in super().modules()] + ["cls"]


class BertLMHeadModel(BertModel):
    def __init__(self, model: nn.Module):
        super().__init__(model, add_pooling_layer=False)

    def modules(self) -> list[str]:
        return [f"bert.{module}" for module in super().modules()] + ["cls"]


class BertForMaskedLM(BertModel):
    def __init__(self, model: nn.Module):
        super().__init__(model, add_pooling_layer=False)

    def modules(self) -> list[str]:
        return [f"bert.{module}" for module in super().modules()] + ["cls"]


class BertForSequenceClassification(BertModel):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        return [f"bert.{module}" for module in super().modules()] + [
            "dropout",
            "classifier",
        ]


class BertForTokenClassification(BertModel):
    def __init__(self, model: nn.Module):
        super().__init__(model, add_pooling_layer=False)

    def modules(self) -> list[str]:
        return [f"bert.{module}" for module in super().modules()] + [
            "dropout",
            "classifier",
        ]


class BertForNextSentencePrediction(BertModel):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        return [f"bert.{module}" for module in super().modules()] + ["cls"]


class BertForMultipleChoice(BertModel):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        return [f"bert.{module}" for module in super().modules()] + [
            "dropout",
            "classifier",
        ]


class BertForQuestionAnswering(BertModel):
    def __init__(self, model: nn.Module):
        super().__init__(model, add_pooling_layer=False)

    def modules(self) -> list[str]:
        return [f"bert.{module}" for module in super().modules()] + ["qa_outputs"]
