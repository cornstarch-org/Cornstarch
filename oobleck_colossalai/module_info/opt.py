from .auto_module import BaseModuleInfo

import torch.nn as nn


class OPTModel(BaseModuleInfo):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        if self.model.__class__.__name__ == "OPTModel":
            module = self.model.decoder
        else:
            module = self.model.model.decoder

        modules = ["decoder.embed_tokens", "decoder.embed_positions"]
        if hasattr(module, "project_in"):
            modules.append("decoder.project_in")

        modules.extend([f"decoder.layers.{i}" for i in range(len(module.layers))])

        if hasattr(module, "project_out"):
            modules.append("decoder.project_out")

        return modules


class OPTForCausalLM(OPTModel):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        return [f"model.{module}" for module in super().modules()] + ["lm_head"]


class OPTForSequenceClassification(OPTModel):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        return [f"model.{module}" for module in super().modules()] + ["score"]


class OPTForQuestionAnswering(OPTModel):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        return [f"model.{module}" for module in super().modules()] + ["qa_outputs"]
