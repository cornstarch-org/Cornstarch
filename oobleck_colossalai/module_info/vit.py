from .auto_module import BaseModuleInfo

import torch.nn as nn


class ViTModel(BaseModuleInfo):
    def __init__(self, model: nn.Module, add_pooling_layer: bool = True):
        super().__init__(model)
        self.add_pooling_layer = add_pooling_layer

    def modules(self) -> list[str]:
        if self.model.__class__.__name__ == "ViTModel":
            module = self.model
        else:
            module = self.model.vit

        modules = (
            ["embeddings"]
            + [f"encoder.layer.{i}" for i in range(len(module.encoder.layer))]
            + ["layernorm"]
        )

        if self.add_pooling_layer:
            modules += ["pooler"]

        return modules


class ViTForImageClassification(ViTModel):
    def __init__(self, model: nn.Module):
        super().__init__(model, add_pooling_layer=False)

    def modules(self) -> list[str]:
        return [f"vit.{module}" for module in super().modules()] + [
            "classifier",
        ]


class ViTForMaskedImageModeling(ViTModel):
    def __init__(self, model: nn.Module):
        super().__init__(model, add_pooling_layer=False)

    def modules(self) -> list[str]:
        return [f"vit.{module}" for module in super().modules()] + [
            "decoder",
        ]
