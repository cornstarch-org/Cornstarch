import torch.nn as nn

from .auto_module import BaseModuleInfo


class T5Model(BaseModuleInfo):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        encoder = self.model.encoder
        decoder = getattr(self.model, "decoder", None)

        num_encoder_layers = len(encoder.block)

        modules = (
            ["shared", "encoder.embed_tokens"]
            + [f"encoder.block.{i}" for i in range(num_encoder_layers)]
            + ["encoder.final_layer_norm", "encoder.dropout"]
        )

        if decoder is not None:
            num_decoder_layers = len(decoder.block)
            modules.extend(
                ["decoder.embed_tokens"]
                + [f"decoder.block.{i}" for i in range(num_decoder_layers)]
                + ["decoder.final_layer_norm", "decoder.dropout"]
            )

        return modules


class T5ForConditionalGeneration(T5Model):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        return super().modules() + ["lm_head"]


class T5EncoderModel(T5Model):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def modules(self) -> list[str]:
        return super().modules()
