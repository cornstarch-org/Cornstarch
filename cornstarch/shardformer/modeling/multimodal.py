import inspect
from typing import Optional

import torch
from transformers.modeling_outputs import BaseModelOutput, ModelOutput

from cornstarch.models.multimodal_language_model.modeling_multimodal_language_model import (
    ModalDecoderModule,
    ModalEncoderModule,
)
from cornstarch.plugin.multimodal_parallel_plugin.multimodal_stage_manager import (
    MultiModalPipelineStageManager,
)


class ModalModulePipelineForwards:
    @staticmethod
    def modal_encoder_module_forward(
        self: ModalEncoderModule,
        stage_manager: MultiModalPipelineStageManager,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> ModelOutput | tuple | dict:
        return_dict = (
            return_dict
            if return_dict is not None
            else self.module.config.use_return_dict
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.module.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.module.config.output_hidden_states
        )

        if stage_manager.is_first_stage(check_only_in_modal=True):
            # Call preprocess callback
            kwargs = self.preprocess_callback(inputs=kwargs)

        # Filter out arguments
        module_params = list(inspect.signature(self.module.forward).parameters.keys())
        module_params = {k: v for k, v in kwargs.items() if k in kwargs}

        outputs: BaseModelOutput | dict | tuple | torch.Tensor = self.module(
            **module_params
        )

        if not stage_manager.is_last_stage(check_only_in_modal=True):
            assert isinstance(outputs, dict), (
                f"Expected the model to return a dictionary, "
                f"but got: {type(outputs)}"
            )
            assert "hidden_states" in outputs.keys(), (
                "Expected the model to return intermediate hidden states, "
                f"but got: {list(outputs.keys())}"
            )
            return outputs

        if isinstance(outputs, torch.Tensor):
            outputs = BaseModelOutput(last_hidden_state=outputs)

        outputs = self.postprocess_module_callback(inputs=kwargs, output=outputs)

        assert isinstance(outputs, (tuple, ModelOutput)), (
            f"Expected the model to return a tuple or ModelOutput, "
            f"but got: {type(outputs)}"
        )

        if self.projector is None:
            return outputs

        outputs = self.projector(outputs[0], return_dict=return_dict)

        # Call postprocess projector callback
        return self.postprocess_projector_callback(inputs=kwargs, output=outputs)

    @staticmethod
    def modal_decoder_module_forward(
        self: ModalDecoderModule,
        stage_manager: MultiModalPipelineStageManager,
        *args,
        **kwargs,
    ) -> ModelOutput | tuple | dict:
        if next(self.projector.parameters(), None) is not None:
            # This rank is responsible for the projector
            return self.module(self.projector(*args, **kwargs))[0]
        else:
            return self.module(*args, **kwargs)
