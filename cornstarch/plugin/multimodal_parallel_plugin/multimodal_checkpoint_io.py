import functools
import logging
import shutil
from pathlib import Path

import torch.distributed as dist
import torch.nn as nn
from colossalai.checkpoint_io import (
    CheckpointIO,
    GeneralCheckpointIO,
    HybridParallelCheckpointIO,
)
from colossalai.checkpoint_io.hybrid_parallel_checkpoint_io import (
    _EXTRA_STATE_KEY_SUFFIX,
)
from colossalai.checkpoint_io.index_file import CheckpointIndexFile
from colossalai.checkpoint_io.utils import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    get_model_base_filenames,
    has_index_file,
    is_safetensors_available,
    load_shard_state_dict,
    load_state_dict_into_model,
    save_config_file,
    save_state_dict_shards,
)
from colossalai.cluster import DistCoordinator
from colossalai.interface import ModelWrapper
from torch.optim import Optimizer
from transformers.modeling_utils import PreTrainedModel

from cornstarch.models.multimodal_language_model.modeling_multimodal_language_model import (
    ModalModuleBase,
)
from cornstarch.plugin.multimodal_parallel_plugin import (
    MultimodalParallelModule,
    MultimodalParallelPlugin,
)
from cornstarch.plugin.multimodal_parallel_plugin.multimodal_stage_manager import (
    MultiModalPipelineStageManager,
)


class ModalParallelCheckpointIO(HybridParallelCheckpointIO):
    def __init__(
        self,
        dp_group: dist.ProcessGroup,
        tp_group: dist.ProcessGroup,
        pp_group: dist.ProcessGroup,
        verbose: bool = True,
    ):
        GeneralCheckpointIO.__init__(self)
        self.dp_group = dp_group
        self.tp_group = tp_group
        self.pp_group = pp_group
        self.dp_rank = dist.get_rank(group=self.dp_group)
        self.tp_rank = dist.get_rank(group=self.tp_group)
        self.pp_rank = dist.get_rank(group=self.pp_group)
        self.verbose = verbose
        self.coordinator = DistCoordinator()

    def clean_index_files(
        self,
        model: PreTrainedModel | ModalModuleBase,
        modal_name: str,
        checkpoint: str,
        stage_manager: MultiModalPipelineStageManager,
        prefix: str | None = None,
        use_safetensors: bool = False,
    ):
        """
        Integrate index files in the temp directory
        """
        if self.dp_rank != 0:
            return

        # Wait until all index files are written.
        dist.barrier(self.pp_group)

        def merge_index_files(tmp_index_file_dir: Path):
            _, save_index_file = get_model_base_filenames(prefix, use_safetensors)

            final_index_file = CheckpointIndexFile(checkpoint)
            final_index_file.append_meta_data("total_size", 0)

            for filename in tmp_index_file_dir.iterdir():
                stage_index_file = CheckpointIndexFile.from_file(
                    tmp_index_file_dir / filename
                )
                final_index_file.metadata["total_size"] += stage_index_file.metadata[
                    "total_size"
                ]
                for weight, weight_filename in stage_index_file.weight_map.items():
                    final_index_file.append_weight_map(weight, weight_filename)

            final_index_file.write_index_file(save_index_file)
            save_config_file(model, checkpoint)

            return save_index_file

        if stage_manager.is_first_stage(check_only_in_modal=True):
            if isinstance(model, PreTrainedModel):
                tmp_index_file_dir = Path(checkpoint) / "tmp_index_files"
                final_index_file = merge_index_files(tmp_index_file_dir)
            elif isinstance(model, ModalModuleBase):
                tmp_index_file_dir = Path(checkpoint)
                final_index_file = merge_index_files(
                    Path(checkpoint) / "module" / "tmp_index_files"
                )
                merge_index_files(Path(checkpoint) / "projector" / "tmp_index_files")
            else:
                raise ValueError(
                    f"model should be an instance of PreTrainedModel or ModalModuleBase, "
                    f"but got {type(model)}."
                )

        dist.barrier(self.pp_group)

        if stage_manager.is_first_stage(check_only_in_modal=True):
            if isinstance(model, PreTrainedModel):
                shutil.rmtree(tmp_index_file_dir)
            elif isinstance(model, ModalModuleBase):
                shutil.rmtree(Path(checkpoint) / "module" / "tmp_index_files")
                shutil.rmtree(Path(checkpoint) / "projector" / "tmp_index_files")

        if self.verbose and stage_manager.is_first_stage(check_only_in_modal=True):
            logging.info(
                f"The model is split into checkpoint shards. "
                f"You can find where each parameters has been saved in the "
                f"index located at {final_index_file}."
            )

    def save_sharded_model(
        self,
        model: PreTrainedModel | ModalModuleBase,
        checkpoint: str,
        gather_dtensor: bool = True,
        prefix: str | None = None,
        size_per_shard: int = 1024,
        use_safetensors: bool = False,
    ) -> None:
        # Devices along the same dp_group share the same copies of model.
        # So only let the device with dp_rank == 0 save the model.
        if self.dp_rank != 0:
            return

        if next(model.parameters(), None) is None:
            return

        Path(checkpoint).mkdir(parents=True, exist_ok=True)

        if isinstance(model, PreTrainedModel):
            if all(not p.requires_grad for p in model.parameters()):
                return

            state_dict_shard = HybridParallelCheckpointIO._model_sharder(
                model, size_per_shard=size_per_shard
            )
            weights_name, save_index_file = get_model_base_filenames(
                prefix, use_safetensors
            )
            index_file = CheckpointIndexFile(checkpoint)
            control_saving = self.tp_rank == 0

            # if stage_manager.num_stages_in_modal == 1:
            #     # Save the model shards as in general CheckpointIO
            #     total_size = save_state_dict_shards(
            #         sharded_state_dict=state_dict_shard,
            #         checkpoint=checkpoint,
            #         index_file=index_file,
            #         base_filename=weights_name,
            #         is_master=control_saving,
            #         use_safetensors=use_safetensors,
            #     )
            #     if control_saving:
            #         index_file.append_meta_data("total_size", total_size)
            #         index_file.write_index_file(save_index_file)
            #         save_config_file(model, checkpoint)
            #         if self.verbose and self.coordinator.is_master():
            #             logging.info(
            #                 f"The model is split into checkpoint shards. "
            #                 f"You can find where each parameters has been saved in the "
            #                 f"index located at {save_index_file}."
            #             )
            # else:
            # When pipeline is used, first each stage produces its own shard files and index files.
            # Index files belonging to each stage are saved under a temporary folder ./tmp_index_files/.
            # After all the state_dicts have been saved, the master rank renames all shard files,
            # integrates all index files into one, and deletes the tmp folder.
            tmp_index_file_dir = Path(checkpoint) / "tmp_index_files"
            tmp_index_file_dir.mkdir(parents=True, exist_ok=True)

            weights_name = weights_name.replace(
                ".bin", f"-stage-{self.pp_rank+1:05d}-shard.bin"
            )
            weights_name = weights_name.replace(
                ".safetensors", f"-stage-{self.pp_rank+1:05d}-shard.safetensors"
            )
            save_index_file = save_index_file.replace(
                ".json", f"-stage-{self.pp_rank+1:05d}-shard.json"
            )
            save_index_file = tmp_index_file_dir / save_index_file

            total_size = save_state_dict_shards(
                sharded_state_dict=state_dict_shard,
                checkpoint=checkpoint,
                index_file=index_file,
                base_filename=weights_name,
                is_master=control_saving,
                use_safetensors=use_safetensors,
                use_pp_format=True,
            )

            if control_saving:
                assert (
                    self.dp_rank == 0 and self.tp_rank == 0
                ), "The saving process should have both dp_rank and tp_rank as 0."
                index_file.append_meta_data("total_size", total_size)
                index_file.write_index_file(save_index_file)

        elif isinstance(model, ModalModuleBase):
            self.save_sharded_model(
                model.module,
                f"{checkpoint}/module",
                gather_dtensor,
                prefix,
                size_per_shard,
                use_safetensors,
            )

            if next(model.projector.parameters(), None) is not None:
                self.save_sharded_model(
                    model.projector,
                    f"{checkpoint}/projector",
                    gather_dtensor,
                    prefix,
                    size_per_shard,
                    use_safetensors,
                )
        else:
            raise ValueError(
                f"model should be an instance of PreTrainedModel or ModalModuleBase, "
                f"but got {type(model)}."
            )

    def load_sharded_model(
        self,
        model: PreTrainedModel | ModalModuleBase,
        checkpoint_index_file: Path,
        strict: bool = False,
    ):
        use_safetensors = False
        if "safetensors" in checkpoint_index_file.name:
            use_safetensors = True

        if use_safetensors and not is_safetensors_available():
            raise ImportError(
                "`safe_serialization` requires the `safetensors` library: `pip install safetensors`."
            )

        if isinstance(model, PreTrainedModel):
            ckpt_index_file = CheckpointIndexFile.from_file(checkpoint_index_file)
            ckpt_root_path = ckpt_index_file.root_path
            weight_map = ckpt_index_file.weight_map
            strict = False

            loaded_files = set()

            missing_keys = []
            missing_file_keys = []

            def load(name: str):
                if name not in weight_map:
                    missing_keys.append(name)
                    return

                file_name = weight_map[name]

                # If this param/buffer has been loaded before, directly return.
                if file_name in loaded_files:
                    return

                file_path = Path(ckpt_root_path) / file_name
                state_dict = load_shard_state_dict(file_path, use_safetensors)

                load_state_dict_into_model(
                    model,
                    state_dict,
                    missing_keys=missing_file_keys,
                    strict=strict,
                    load_sub_module=True,
                )
                loaded_files.add(file_name)

            # Load parameters
            for name, _ in model.named_parameters():
                load(name)

            # Load buffers
            non_persistent_buffers = set()
            for name, module in model.named_modules():
                non_persistent_buffers |= set(
                    ".".join((name, b)) for b in module._non_persistent_buffers_set
                )
            for name, buf in model.named_buffers():
                if buf is not None and name not in non_persistent_buffers:
                    load(name)

            # Load extra states
            extra_state_key = _EXTRA_STATE_KEY_SUFFIX
            if (
                getattr(model.__class__, "get_extra_state", nn.Module.get_extra_state)
                is not nn.Module.get_extra_state
            ):
                load(extra_state_key)

            # TODO: update master params if mixed-precision training is enabled

            if self.verbose and self.coordinator.is_master():
                logging.info(
                    f"The model has been successfully loaded from sharded checkpoint: {ckpt_root_path}."
                )

            if len(missing_keys) == 0:
                raise RuntimeError(
                    "No weigth is loaded into the model. Please check the checkpoint files and the model structure."
                )

            remain_keys = functools.reduce(lambda a, b: a & b, map(set, missing_keys))
            remain_keys = remain_keys.union(set(missing_file_keys))
            if len(remain_keys) > 0:
                if strict:
                    error_msgs = "Missing key(s) in state_dict: {}. ".format(
                        ", ".join('"{}"'.format(k) for k in missing_keys)
                    )
                    raise RuntimeError(
                        "Error(s) in loading state_dict for {}:\n\t{}".format(
                            self.__class__.__name__, "\n\t".join(error_msgs)
                        )
                    )
                else:
                    if self.coordinator.is_master():
                        logging.info(
                            f"The following keys are not loaded from checkpoint: {remain_keys}"
                        )

        elif isinstance(model, ModalModuleBase):
            pass
        else:
            raise ValueError(
                f"model should be an instance of PreTrainedModel or ModalModuleBase, "
                f"but got {type(model)}."
            )

    def save_unsharded_model(
        self,
        model: ModelWrapper,
        checkpoint: str,
        gather_dtensor: bool,
        use_safetensors: bool,
    ):
        raise NotImplementedError


class MultimodalParallelCheckpointIO(CheckpointIO):
    """
    Multimodal CheckpointIO class that stores multiple modal modules in a hierarchical structure.

    Example of vision language model checkpoint structure:
    - checkpoint
        - vision_encoder
            - module
                - model.pt (unsharded) or model-0000x-of-0000y.pt (sharded)
                - model.index.json (sharded)
            - projector
                - model.pt (unsharded) or model-0000x-of-0000y.pt (sharded)
                - model.index.json (sharded)
        - language_model
            - model.pt (unsharded) or model-0000x-of-0000y.pt (sharded)
            - model.index.json (sharded)
    """

    def __init__(self, plugin: MultimodalParallelPlugin):
        super().__init__()
        self.plugin = plugin

    def save_model(
        self,
        model: MultimodalParallelModule,
        checkpoint: str,
        shard: bool = False,
        gather_dtensor: bool = True,
        prefix: str = None,
        size_per_shard: int = 1024,
        use_safetensors: bool = False,
    ):
        """
        Save model to checkpoint.

        Each modal is saved hierarchically following its model structure under `checkpoint` path
        as described in the class docstring.
        For this, `checkpoint` should be a directory.

        If a module is frozen, it will not be saved.
        Whether the module is not frozen is determined by if any parameter in the module has `requires_grad` set to `True`.

        Args:
            model (MultimodalParallelModule): a multimodal parallel model to save.
            checkpoint (str): a directory path to save the model. It should be a directory path, not a file.
                The directory path doesn't have to exist.
            shard (bool): whether to save the sharded checkpoint.
                Each modal module will be sharded into multiple files.
                The model shards will be specified by a `model.index.json` file.
            gather_dtensor (bool): whether to gather the distributed tensor to the first device. Default: True.
            prefix (str): If specified, weights are saved in the format pytorch_model.<prefix>.bin. Default: None.
                This value is only used when shard = True.
            size_per_shard (int): size per shard in MB. Default: 1024.
                This value is only used when shard = True.
            use_safetensors (bool): whether to use safe tensors. Default: False.
                If set to True, the checkpoint will be saved in .safetensor format.
        """
        assert isinstance(model, MultimodalParallelModule), (
            f"model should be an instance of MultimodalParallelModule, "
            f"but got {type(model)}."
        )

        checkpoint: Path = Path(checkpoint)
        assert (
            not checkpoint.suffix
        ), "checkpoint path should be a directory for multimodal model."
        checkpoint.mkdir(parents=True, exist_ok=True)

        module = getattr(model.module, model.my_modal_name)
        checkpoint_io = ModalParallelCheckpointIO(
            self.plugin.dp_group, self.plugin.tp_group, self.plugin.pp_groups[0]
        )

        if shard:
            checkpoint_name = f"{str(checkpoint)}/{model.my_modal_name}"
            checkpoint_io.save_sharded_model(
                module,
                checkpoint_name,
                gather_dtensor,
                prefix,
                size_per_shard,
                use_safetensors,
            )

            checkpoint_io.clean_index_files(
                module,
                model.my_modal_name,
                checkpoint_name,
                self.plugin.stage_manager,
                prefix,
                use_safetensors,
            )
        else:
            checkpoint_io.save_unsharded_model(
                module,
                f"{str(checkpoint)}/{model.my_modal_name}",
                gather_dtensor,
                use_safetensors,
                self.plugin.stage_manager,
            )

    def load_model(
        self, model: MultimodalParallelModule, checkpoint: str, strict: bool = True
    ) -> MultimodalParallelModule:
        assert isinstance(model, MultimodalParallelModule), (
            f"model should be an instance of MultimodalParallelModule, "
            f"but got {type(model)}."
        )
        original_model = model

        # since we only support loaded sharded and unsharded weight format
        # containing no distributed tensors, dtensor -> full tensor conversion
        # should be done offline via our CLI
        # the existence of index file means it is a sharded checkpoint
        index_file_exists, index_file_path = has_index_file(checkpoint)

        if index_file_exists:
            self.load_sharded_model(model, index_file_path, strict)
        else:
            raise NotImplementedError()
            path = Path(checkpoint, SAFE_WEIGHTS_NAME)
            if path.is_file():
                self.load_unsharded_model(model, str(path), strict)
            else:
                path = Path(checkpoint, WEIGHTS_NAME)
                if path.is_file():
                    self.load_unsharded_model(model, str(path), strict)
                else:
                    self.load_unsharded_model(model, checkpoint, strict)

        return original_model

    def load_sharded_model(self, model: nn.Module, index_file_path: str, strict: bool):
        raise NotImplementedError("Not used in MultimodalParallelCheckpointIO")

    def load_sharded_optimizer(
        self, optimizer: Optimizer, index_file_path: str, prefix: str
    ):
        raise NotImplementedError("Not used in MultimodalParallelCheckpointIO")

    def load_unsharded_model(self, model: nn.Module, checkpoint: str, strict: bool):
        raise NotImplementedError("Not used in MultimodalParallelCheckpointIO")

    def load_unsharded_optimizer(
        self, optimizer: Optimizer, checkpoint: str, strict: bool
    ):
        raise NotImplementedError("Not used in MultimodalParallelCheckpointIO")

    def save_lora_as_pretrained(
        self,
        model: nn.Module | ModelWrapper,
        checkpoint: str,
        use_safetensors: bool = False,
    ) -> None:
        raise NotImplementedError("TODO: implement it")

    def save_sharded_model(
        self,
        model: nn.Module,
        checkpoint: str,
        gather_dtensor: bool,
        prefix: str | None,
        size_per_shard: int,
        use_safetensors: bool,
    ):
        raise NotImplementedError("Not used in MultimodalParallelCheckpointIO")

    def save_sharded_optimizer(
        self,
        optimizer: Optimizer,
        checkpoint: Path,
        gather_dtensor: bool,
        prefix: str,
        size_per_shard: int,
    ):
        raise NotImplementedError("Not used in MultimodalParallelCheckpointIO")

    def save_unsharded_model(
        self,
        model: nn.Module,
        checkpoint: str,
        gather_dtensor: bool,
        use_safetensors: bool,
    ):
        raise NotImplementedError("Not used in MultimodalParallelCheckpointIO")

    def save_unsharded_optimizer(
        self, optimizer: Optimizer, checkpoint: Path, gather_dtensor: bool
    ):
        raise NotImplementedError("Not used in MultimodalParallelCheckpointIO")
