import copy
import functools
import logging
import shutil
from pathlib import Path
from typing import Optional

import torch
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
    get_optimizer_base_filenames,
    has_index_file,
    is_safetensors_available,
    load_shard_state_dict,
    load_state_dict,
    load_state_dict_into_model,
    load_states_into_optimizer,
    save_config_file,
    save_param_groups,
    save_state_dict_shards,
    sharded_optimizer_loading_epilogue,
)
from colossalai.cluster import DistCoordinator
from colossalai.interface import ModelWrapper, OptimizerWrapper
from torch.optim import Optimizer
from transformers.modeling_utils import PreTrainedModel

from cornstarch.models.multimodal_language_model.modeling_multimodal_language_model import (
    ModalEncoderModule,
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
        self.tp_size = dist.get_world_size(group=self.tp_group)
        self.use_zero = False
        self.verbose = verbose
        self.coordinator = DistCoordinator()

    def clean_model_index_files(
        self,
        model: PreTrainedModel | ModalModuleBase,
        checkpoint: str,
        stage_manager: MultiModalPipelineStageManager,
        prefix: str | None = None,
        use_safetensors: bool = False,
    ):
        """
        Integrate index files in the temp directory to one final index file.
        """
        if self.dp_rank != 0 or self.tp_rank != 0:
            return

        # Wait until all index files are written.
        dist.barrier(self.pp_group)

        def merge_index_files(module: PreTrainedModel, tmp_index_file_dir: Path):
            _, save_index_file = get_model_base_filenames(prefix, use_safetensors)

            final_index_file = CheckpointIndexFile(str(tmp_index_file_dir.parent))
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
            save_config_file(module, str(tmp_index_file_dir.parent))

            return save_index_file

        if isinstance(model, PreTrainedModel):
            if (
                stage_manager.is_first_stage(check_only_in_modal=True)
                and model.training
            ):
                merge_index_files(model, Path(checkpoint) / "tmp_index_files")
        elif isinstance(model, ModalEncoderModule):
            if (
                stage_manager.is_first_stage(check_only_in_modal=True)
                and model.module.training
            ):
                merge_index_files(
                    model.module,
                    Path(checkpoint) / "module" / "tmp_index_files",
                )
            if (
                stage_manager.is_last_stage(check_only_in_modal=True)
                and model.projector.training
            ):
                merge_index_files(
                    model.projector,
                    Path(checkpoint) / "projector" / "tmp_index_files",
                )
        else:
            raise ValueError(
                f"model should be an instance of PreTrainedModel or ModalModuleBase, "
                f"but got {type(model)}."
            )

        dist.barrier(self.pp_group)

        if stage_manager.is_first_stage(check_only_in_modal=True):
            if isinstance(model, PreTrainedModel):
                shutil.rmtree(
                    Path(checkpoint) / "tmp_index_files",
                    ignore_errors=True,
                )
            elif isinstance(model, ModalModuleBase):
                shutil.rmtree(
                    Path(checkpoint) / "module" / "tmp_index_files",
                    ignore_errors=True,
                )
                shutil.rmtree(
                    Path(checkpoint) / "projector" / "tmp_index_files",
                    ignore_errors=True,
                )

    def clean_optimizer_index_files(
        self,
        optimizer: OptimizerWrapper,
        checkpoint: str,
        stage_manager: MultiModalPipelineStageManager,
        prefix: str | None = None,
    ):
        """
        Integrate index files in the temp directory to one final index file.
        """
        if self.dp_rank != 0 or self.tp_rank != 0:
            return

        # Wait until all index files are written.
        dist.barrier(self.pp_group)

        def merge_index_files(tmp_index_file_dir: Path):
            _, save_index_file, param_group_file = get_optimizer_base_filenames(prefix)

            final_index_file = CheckpointIndexFile(str(tmp_index_file_dir.parent))
            final_index_file.append_meta_data("total_size", 0)

            for filename in tmp_index_file_dir.iterdir():
                stage_index_file = CheckpointIndexFile.from_file(
                    tmp_index_file_dir / filename
                )
                final_index_file.metadata["total_size"] += stage_index_file.metadata[
                    "total_size"
                ]
                for param_id, state_filename in stage_index_file.weight_map.items():
                    final_index_file.append_weight_map(param_id, state_filename)

            # Store param groups.
            final_index_file.append_meta_data("param_groups", param_group_file)
            group_file_path = Path(checkpoint) / param_group_file
            param_groups = [
                {**group, "params": group_info["params"]}
                for group, group_info in zip(
                    optimizer.param_groups, optimizer.param_info["param_groups"]
                )
            ]
            save_param_groups({"param_groups": param_groups}, group_file_path)

            final_index_file.write_index_file(save_index_file)

            return save_index_file

        if stage_manager.is_first_stage(check_only_in_modal=True):
            tmp_index_file_dir = Path(checkpoint) / "tmp_index_files"
            final_index_file = merge_index_files(tmp_index_file_dir)

        dist.barrier(self.pp_group)

        if stage_manager.is_first_stage(check_only_in_modal=True):
            shutil.rmtree(tmp_index_file_dir)

            if self.verbose:
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

        if not model.training:
            return

        Path(checkpoint).mkdir(parents=True, exist_ok=True)

        if isinstance(model, PreTrainedModel):
            state_dict_shard = HybridParallelCheckpointIO._model_sharder(
                model, size_per_shard=size_per_shard
            )
            weights_name, save_index_file = get_model_base_filenames(
                prefix, use_safetensors
            )
            index_file = CheckpointIndexFile(checkpoint)
            control_saving = self.tp_rank == 0

            # Each stage produces its own shard files and index files.
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
        model: PreTrainedModel,
        checkpoint_index_file: Path,
        strict: bool = False,
    ):
        assert isinstance(model, PreTrainedModel), (
            f"model should be an instance of PreTrainedModel, "
            f"but got {type(model)}."
        )

        use_safetensors = False
        if "safetensors" in checkpoint_index_file.name:
            use_safetensors = True

        if use_safetensors and not is_safetensors_available():
            raise ImportError(
                "`safe_serialization` requires the `safetensors` library: `pip install safetensors`."
            )

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

        if self.verbose and self.coordinator.is_master():
            logging.info(
                f"The model has been successfully loaded from sharded checkpoint: {ckpt_root_path}."
            )

        if len(missing_keys) == 0:
            # No weights is loaded. This is likely because modules are separated.
            return

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

    def load_unsharded_model(
        self, model: PreTrainedModel, checkpoint: str, strict: bool = False
    ):
        assert isinstance(model, PreTrainedModel), (
            f"model should be an instance of PreTrainedModel, "
            f"but got {type(model)}."
        )

        state_dict = load_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=strict)

    def save_unsharded_model(
        self,
        model: ModelWrapper,
        checkpoint: str,
        gather_dtensor: bool,
        use_safetensors: bool,
    ):
        raise NotImplementedError

    def save_sharded_optimizer(
        self,
        optimizer: OptimizerWrapper,
        checkpoint: str,
        gather_dtensor: bool = True,
        prefix: str | None = None,
        size_per_shard: int = 1024,
    ):
        # Devices along the same dp_group share the same copies of states.
        # So only let the device with dp_rank == 0 save the states.
        if self.dp_rank != 0:
            return

        # Collect the sharded states along tp_group.
        # Only devices with (dp_rank == 0 and tp_rank == 0) are responsible for states saving.
        state_dict_shard = HybridParallelCheckpointIO._optimizer_sharder(
            optimizer,
            use_zero=self.use_zero,
            dp_group=self.dp_group,
            tp_group=self.tp_group,
            size_per_shard=size_per_shard,
        )
        states_name, save_index_file, _ = get_optimizer_base_filenames(prefix)
        index_file = CheckpointIndexFile(checkpoint)
        control_saving = self.tp_rank == 0

        # Each stage produces its own shard files and index files.
        # Index files belonging to each stage are saved under a temporary folder ./tmp_index_files/.
        # After all the state_dicts have been saved, the master rank integrates all the index files into one.
        tmp_index_file_dir = Path(checkpoint) / "tmp_index_files"
        tmp_index_file_dir.mkdir(parents=True, exist_ok=True)

        states_name = states_name.replace(
            ".bin", f"-stage-{self.pp_rank+1:05d}-shard.bin"
        )
        save_index_file = save_index_file.replace(
            ".json", f"-stage-{self.pp_rank+1:05d}-shard.json"
        )
        save_index_file = tmp_index_file_dir / save_index_file

        total_size = save_state_dict_shards(
            sharded_state_dict=state_dict_shard,
            checkpoint=checkpoint,
            index_file=index_file,
            base_filename=states_name,
            is_master=control_saving,
            use_safetensors=False,
            use_pp_format=True,
        )

        if control_saving:
            assert (
                self.dp_rank == 0 and self.tp_rank == 0
            ), "The saving process should have both dp_rank and tp_rank as 0."
            index_file.append_meta_data("total_size", total_size)
            index_file.write_index_file(save_index_file)

    def load_sharded_optimizer(
        self, optimizer: OptimizerWrapper, checkpoint_index_file: str, prefix: str = ""
    ):
        assert isinstance(
            optimizer, OptimizerWrapper
        ), "Please boost the optimizer before loading!"

        def get_param_id_from_optimizer_param(
            param: torch.Tensor,
            master_to_working_map: Optional[dict[int, torch.Tensor]] = None,
        ):
            if master_to_working_map is not None:
                working_param = master_to_working_map[id(param)]
            else:
                working_param = param
            return optimizer.param_info["param2id"][id(working_param)]

        # id_map is a mapping from param ids kept by current pipeline, to their corresponding parameter objects.
        id_map: dict[int, torch.Tensor] = {}
        master_to_working_map = optimizer.get_master_to_working_map()
        for pg in optimizer.optim.param_groups:
            for param in pg["params"]:
                param_id = get_param_id_from_optimizer_param(
                    param, master_to_working_map
                )
                id_map[param_id] = param

        # Read checkpoint index file.
        ckpt_index_file = CheckpointIndexFile.from_file(checkpoint_index_file)
        ckpt_root_path = ckpt_index_file.root_path
        weight_map = ckpt_index_file.weight_map
        weight_map = {
            int(k): v for k, v in weight_map.items()
        }  # convert saved id from str to int

        # Load param groups
        param_group_path = ckpt_index_file.get_param_group_filename()
        if param_group_path is None:
            raise RuntimeError(
                f"Invalid index file path {checkpoint_index_file} for an optimizer. \
                               Lacking param group file under current directory."
            )

        saved_groups = torch.load(param_group_path)

        updated_groups = []
        for old_pg, saved_pg in zip(optimizer.optim.param_groups, saved_groups):
            # obtain updated param group
            new_pg = copy.deepcopy(saved_pg)
            new_pg["params"] = old_pg[
                "params"
            ]  # The parameters in the same group shouldn't change.
            updated_groups.append(new_pg)
        optimizer.optim.__dict__.update({"param_groups": updated_groups})

        # Load saved states to optimizer.
        # Keep a record of loaded files so that file will not be repeatedly loaded.
        loaded_file = set()
        for pg in optimizer.optim.param_groups:
            for param in pg["params"]:
                if param is None:
                    continue
                param_id = get_param_id_from_optimizer_param(
                    param, master_to_working_map
                )
                if param_id not in weight_map:
                    continue
                filename = weight_map[param_id]

                # If this param's states has been loaded before, directly return.
                if filename in loaded_file:
                    continue

                file_path = ckpt_root_path / filename
                state_dict = load_shard_state_dict(
                    Path(file_path), use_safetensors=False
                )
                load_states_into_optimizer(
                    optimizer.optim, state_dict, id_map, strict=True
                )
                loaded_file.add(filename)

        # Sharding and epiloge is done from multimodalcheckpointio

        if self.verbose and self.coordinator.is_master():
            logging.info(
                f"The optimizer has been successfully loaded from sharded checkpoint: {ckpt_root_path}."
            )


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
        For this, `checkpoint` should be a dictionary.

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
        checkpoint.mkdir(parents=True, exist_ok=True)
        assert (
            checkpoint.is_dir()
        ), "checkpoint path should be a directory for multimodal model."

        module = getattr(model.module, model.my_modal_name)
        checkpoint_io = ModalParallelCheckpointIO(
            self.plugin.dp_group, self.plugin.tp_group, self.plugin.pp_groups[0]
        )

        checkpoint_name = f"{str(checkpoint)}/{model.my_modal_name}"
        if shard:
            checkpoint_io.save_sharded_model(
                module,
                checkpoint_name,
                gather_dtensor,
                prefix,
                size_per_shard,
                use_safetensors,
            )

            checkpoint_io.clean_model_index_files(
                module,
                checkpoint_name,
                self.plugin.stage_manager,
                prefix,
                use_safetensors,
            )
        else:
            checkpoint_io.save_unsharded_model(
                module,
                checkpoint_name,
                gather_dtensor,
                use_safetensors,
                self.plugin.stage_manager,
            )

    def load_model(
        self,
        model: MultimodalParallelModule,
        checkpoint: dict[str, str],
        strict: bool = True,
    ) -> MultimodalParallelModule:
        """
        Load model modules from checkpoints.

        Each modality module can be loaded individually from the corresponding checkpoint path.
        Example of vision language model checkpoint structure:
        checkpoint = {
            "language_model": "path/to/language_model",
            "vision_encoder.module": "path/to/vision_encoder/module",
            "vision_encoder.projector": "path/to/vision_encoder/projector",
        }

        A vision encoder and a projector can be loaded separately, as they are individual modules.
        If a checkpoint includes multiple modules, same path can be used for multiple modules.

        If a model is lazy initialized with `from_pretrained()`, the path is already stored in the model.
        The path given to `from_pretrained()` will be used and the corresponding path in the checkpoint dictionary will be ignored.
        """
        assert isinstance(model, MultimodalParallelModule), (
            f"model should be an instance of MultimodalParallelModule, "
            f"but got {type(model)}."
        )
        original_model = model

        def load_module(module: nn.Module, target_index_file_path: Path):
            # since we only support loaded sharded and unsharded weight format
            # containing no distributed tensors, dtensor -> full tensor conversion
            # should be done offline via our CLI
            # the existence of index file means it is a sharded checkpoint
            index_file_exists, index_file_path = has_index_file(
                str(target_index_file_path)
            )

            checkpoint_io = ModalParallelCheckpointIO(
                self.plugin.dp_group, self.plugin.tp_group, self.plugin.pp_groups[0]
            )

            if index_file_exists:
                checkpoint_io.load_sharded_model(module, index_file_path, strict)
            else:
                path = Path(target_index_file_path, SAFE_WEIGHTS_NAME)
                if path.is_file():
                    checkpoint_io.load_unsharded_model(module, str(path), strict)
                else:
                    path = Path(target_index_file_path, WEIGHTS_NAME)
                    if path.is_file():
                        checkpoint_io.load_unsharded_model(module, str(path), strict)
                    else:
                        checkpoint_io.load_unsharded_model(
                            module, str(target_index_file_path), strict
                        )

        def get_checkpoint_path(
            module: PreTrainedModel | ModalModuleBase, module_name: str
        ) -> Optional[Path]:
            path = getattr(module, "_pretrained", None)
            if path is None and checkpoint is not None:
                path = checkpoint.get(module_name, None)

            if path is None:
                if strict:
                    raise ValueError(
                        f"checkpoint path for {module_name} is not provided in the checkpoint dictionary."
                    )

                return None

            return Path(path)

        language_model = getattr(model.module, "language_model")
        language_model_path = get_checkpoint_path(language_model, "language_model")
        if language_model_path is not None:
            load_module(language_model, language_model_path)

        for modal_key, encoder_module in model.module.encoders.items():
            assert isinstance(encoder_module, ModalEncoderModule), (
                f"encoder_module should be an instance of ModalEncoderModule, "
                f"but got {type(encoder_module)}."
            )

            encoder_module_path = get_checkpoint_path(
                encoder_module.module, f"{modal_key}.module"
            )
            if encoder_module_path is not None:
                load_module(encoder_module.module, Path(encoder_module_path))

            projector_path = get_checkpoint_path(
                encoder_module.projector, f"{modal_key}.projector"
            )
            if projector_path is not None:
                load_module(encoder_module.projector, Path(projector_path))

        # Update master params if mixed-precision training is enabled.
        model.update_master_params()

        return original_model

    def save_optimizer(
        self,
        optimizer: OptimizerWrapper,
        checkpoint: str,
        shard: bool = False,
        gather_dtensor=True,
        prefix: str = None,
        size_per_shard: int = 1024,
    ):
        assert isinstance(
            optimizer, OptimizerWrapper
        ), "Please boost the optimizer before saving!"
        checkpoint: Path = Path(checkpoint)
        assert (
            not checkpoint.suffix
        ), "checkpoint path should be a directory for multimodal model."
        checkpoint.mkdir(parents=True, exist_ok=True)

        checkpoint_io = ModalParallelCheckpointIO(
            self.plugin.dp_group, self.plugin.tp_group, self.plugin.pp_groups[0]
        )

        checkpoint_name = f"{str(checkpoint)}/{optimizer.model.my_modal_name}"
        if shard:
            checkpoint_io.save_sharded_optimizer(
                optimizer,
                checkpoint_name,
                gather_dtensor,
                prefix,
                size_per_shard,
            )

            checkpoint_io.clean_optimizer_index_files(
                optimizer,
                checkpoint_name,
                self.plugin.stage_manager,
                prefix,
            )
        else:
            checkpoint_io.save_unsharded_optimizer(
                optimizer,
                checkpoint_name,
                gather_dtensor,
            )

    def load_optimizer(
        self,
        optimizer: OptimizerWrapper,
        checkpoint: dict[str, str],
        prefix: str = None,
        size_per_shard: int = 1024,
    ):
        """
        Load optimizer from checkpoint.

        Each modal is saved hierarchically following its model structure under `checkpoint` path
        as described in the class docstring.
        For this, `checkpoint` should be a dictionary.

        Unlike `load_model` method that loads a module and a projector for each modality separately,
        optimizer can't be loaded separately for each modality since it is stored as one checkpoint.

        Example of vision language model checkpoint structure:
        checkpoint = {
            "language_model": "path/to/language_model",
            "vision_encoder": "path/to/vision_encoder",
        }
        """
        assert hasattr(
            optimizer, "model"
        ), "optimizer should hold a module, named model."
        assert isinstance(getattr(optimizer, "model"), MultimodalParallelModule), (
            f"optimizer should hold a module typed MultimodalParallelModule, "
            f"but got {type(getattr(optimizer, 'model'))}."
        )

        checkpoint_io = ModalParallelCheckpointIO(
            self.plugin.dp_group, self.plugin.tp_group, self.plugin.pp_groups[0]
        )

        def load_optimizer_states(
            optimizer: OptimizerWrapper, target_index_file_path: Path
        ):
            index_file_exists, index_file_path = has_index_file(
                str(target_index_file_path)
            )

            if index_file_exists:
                checkpoint_io.load_sharded_optimizer(optimizer, index_file_path, prefix)
            else:
                checkpoint_io.load_unsharded_optimizer(optimizer, checkpoint)

        if "language_model" in checkpoint:
            load_optimizer_states(optimizer, Path(checkpoint.pop("language_model")))

        for encoder_name in optimizer.model.module.encoders.keys():
            if f"{encoder_name}_encoder" in checkpoint:
                load_optimizer_states(
                    optimizer, Path(checkpoint.pop(f"{encoder_name}_encoder"))
                )

        # Shard the loaded optimizer states if using tp.
        master_to_working_map = optimizer.get_master_to_working_map()
        for i, (param, state) in enumerate(optimizer.optim.state.items()):
            device = param.device
            if master_to_working_map is not None:
                working_param = master_to_working_map[id(param)]
            else:
                working_param = param

            original_shape = optimizer.param_info["param2shape"][id(working_param)]
            sharded_state = checkpoint_io.shard_from_complete_optimizer_state(
                state,
                current_shape=working_param.shape,
                original_shape=original_shape,
                device=device,
                inplace=True,
            )
            optimizer.optim.state[param] = sharded_state

        sharded_optimizer_loading_epilogue(optimizer.optim)

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
