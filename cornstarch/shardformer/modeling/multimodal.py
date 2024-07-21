from typing import Optional

import torch
from transformers.modeling_outputs import ModelOutput

from cornstarch.models.multimodal_language_model.modeling_multimodal_language_model import (
    ModalModule,
    ModalModuleType,
)
from cornstarch.plugin.multimodal_parallel_plugin.multimodal_stage_manager import (
    MultiModalPipelineStageManager,
)


class ModalModulePipelineForwards:
    @staticmethod
    def modal_module_forward(
        self: ModalModule,
        stage_manager: MultiModalPipelineStageManager,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if self.projector is None:
            return self.module(*args, **kwargs)

        if self.modal_type == ModalModuleType.Encoder:
            module_output = self.module(*args, **kwargs)
            if isinstance(module_output, (tuple, ModelOutput)):
                assert stage_manager.is_last_stage(
                    check_only_in_modal=True
                ), "Intermediate stage should not return final output."
                assert (
                    next(self.projector.parameters(), None) is not None
                ), "Projector parameters are released while the model returns its final output."
                return self.projector(module_output[0])
            else:
                # Module returns intermediate outputs with pipeline parallelism
                assert not stage_manager.is_last_stage(
                    check_only_in_modal=True
                ), "Last stage should not return intermediate hidden states."
                assert (
                    isinstance(module_output, dict)
                    and "hidden_states" in module_output.keys()
                ), (
                    "Expected the model to return intermediate hidden states, "
                    f"but got: {list(module_output.keys())}"
                )
                assert (
                    next(self.projector.parameters(), None) is None
                ), "Projector parameters are not released while the model returns intermediate outputs."
                return module_output
        else:
            if next(self.projector.parameters(), None) is not None:
                # This rank is responsible for the projector
                return self.module(self.projector(*args, **kwargs))[0]
            else:
                return self.module(*args, **kwargs)
