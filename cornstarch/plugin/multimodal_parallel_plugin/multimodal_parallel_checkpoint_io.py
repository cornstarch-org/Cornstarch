from types import MethodType

import torch.distributed as dist
from colossalai.checkpoint_io import GeneralCheckpointIO, HybridParallelCheckpointIO
from colossalai.cluster import DistCoordinator
from colossalai.interface import ModelWrapper

from cornstarch.models.multimodal_language_model import MultimodalModel


class MultimodalParallelCheckpointIO(HybridParallelCheckpointIO):
    """
    CheckpointIO for Multimodal Language Model.
    """

    def __init__(
        self,
        dp_group: dist.ProcessGroup = None,
        pp_group: dist.ProcessGroup = None,
        tp_group: dist.ProcessGroup = None,
        verbose: bool = True,
    ):
        GeneralCheckpointIO.__init__(self)
        self.dp_group = dp_group
        self.pp_group = pp_group
        self.tp_group = tp_group
        self.verbose = verbose
        self.coordinator = DistCoordinator()

    def load_model(
        self,
        model: ModelWrapper,
        checkpoint: dict[str, str],
        strict: bool = False,
    ) -> ModelWrapper:
        """
        Load model from checkpoints.

        MultimodalModel includes multiple modalities, each of which has
        its own checkpoint. This function loads the checkpoint of each modality
        and initializes the corresponding model.
        """

        assert isinstance(model, ModelWrapper), "Please boost the model before loading!"

        # return the original model instead of the unwrapped model
        origin_model = model

        model: MultimodalModel = model.unwrap()
        assert isinstance(
            model, MultimodalModel
        ), "Wrapped model must be MultimodalModel!"

        for key, checkpoint_path in checkpoint.items():
            if checkpoint_path is None:
                continue

            module = model.get_submodule(key)

            # Fake checkpointIO that each modality is a separate model.
            # This is to avoid the error due to that the checkpointIO
            # assumes the given model is boosted.
            module = ModelWrapper(module)
            module.update_master_params = MethodType(lambda *args: None, model)
            super().load_model(module, checkpoint_path, strict)

        # Update master params if mixed-precision training is enabled.
        origin_model.update_master_params()

        return origin_model
