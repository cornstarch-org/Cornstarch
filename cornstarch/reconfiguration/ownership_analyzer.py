"""Analyzer for determining tensor ownership across ranks."""

from typing import Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn

from .data_structures import LayerOwnership
from .utils import get_parent_module


class TensorOwnershipAnalyzer:
    """Analyzes current layer ownership in a parallelized model."""

    def __init__(self, model: nn.Module, plugin=None):
        """Initialize the analyzer.

        Args:
            model: The parallelized model
            plugin: The parallel plugin (optional, for future use)
        """
        self.model = model
        self.plugin = plugin
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def analyze(
        self,
        tp_group: "Optional[dist.ProcessGroup]" = None,
    ) -> Dict[int, LayerOwnership]:
        """Analyze ownership across all ranks.

        Args:
            tp_group: Current TP process group.  When provided, TP-sharded
                parameters (``Linear1D_Col`` / ``Linear1D_Row``) are annotated
                with the range they hold in the full parameter tensor.  When
                omitted (or when the group has size 1) every owned parameter is
                treated as a complete (un-sharded) tensor.

        Returns:
            Dictionary mapping rank to LayerOwnership.
        """
        local_ownership = self._get_local_ownership(tp_group=tp_group)
        return self._gather_ownership(local_ownership)

    def _get_local_ownership(
        self,
        tp_group: "Optional[dist.ProcessGroup]" = None,
    ) -> LayerOwnership:
        """Get layers held by this rank, annotating TP shard ranges when applicable."""
        try:
            from colossalai.shardformer.layer import Linear1D_Col, Linear1D_Row
            _tp_classes = (Linear1D_Col, Linear1D_Row)
        except ImportError:
            _tp_classes = ()

        # Current TP rank / size (for detecting shard ranges).
        tp_rank = (
            dist.get_rank(tp_group)
            if tp_group is not None and dist.get_world_size(tp_group) > 1
            else 0
        )
        tp_size = (
            dist.get_world_size(tp_group)
            if tp_group is not None and dist.get_world_size(tp_group) > 1
            else 1
        )

        # Build param_full_name → immediate parent module mapping.
        param_to_module: Dict[str, "nn.Module"] = {}
        for mod_name, mod in self.model.named_modules():
            for local_name, _ in mod.named_parameters(recurse=False):
                full = f"{mod_name}.{local_name}" if mod_name else local_name
                param_to_module[full] = mod

        layer_names: List[str] = []
        is_placeholder: Dict[str, bool] = {}
        shard_range: Dict[str, Optional[tuple]] = {}
        shard_dim: Dict[str, Optional[int]] = {}

        for name, param in self.model.named_parameters():
            if self._is_placeholder(name, param):
                is_placeholder[name] = True
                shard_range[name] = None
                shard_dim[name] = None
                continue

            layer_names.append(name)
            is_placeholder[name] = False

            # Detect TP sharding.
            mod = param_to_module.get(name)
            local_name = name.split(".")[-1]
            if _tp_classes and mod is not None and isinstance(mod, _tp_classes) and tp_size > 1:
                if local_name == "weight":
                    dim = 0 if isinstance(mod, _tp_classes[0]) else 1
                elif local_name == "bias" and isinstance(mod, _tp_classes[0]):
                    dim = 0
                else:
                    shard_range[name] = None
                    shard_dim[name] = None
                    continue
                local_size = param.shape[dim]
                shard_range[name] = (tp_rank * local_size, (tp_rank + 1) * local_size)
                shard_dim[name] = dim
            else:
                shard_range[name] = None
                shard_dim[name] = None

        return LayerOwnership(
            rank=self.rank,
            layer_names=layer_names,
            is_placeholder=is_placeholder,
            shard_range=shard_range,
            shard_dim=shard_dim,
        )

    def _is_placeholder(self, name: str, param) -> bool:
        """Check if a parameter is a TensorPlaceholder.

        Args:
            name: Parameter name
            param: Parameter value

        Returns:
            True if parameter is a placeholder
        """
        # Import here to avoid circular dependencies
        from cornstarch.shardformer.shard.placeholder import TensorPlaceholder

        # Method 1: Check type directly
        if isinstance(param, TensorPlaceholder):
            return True

        # Method 2: Check if parameter is None (after placeholder conversion)
        if param is None:
            return True

        # Method 3: Check module's _parameter_placeholders attribute
        try:
            parent_module = get_parent_module(self.model, name)
            if hasattr(parent_module, '_parameter_placeholders'):
                param_local_name = name.split('.')[-1]
                if param_local_name in parent_module._parameter_placeholders:
                    return True
        except Exception:
            # If we can't find the parent module, assume it's not a placeholder
            pass

        return False

    def _gather_ownership(
        self,
        local_ownership: LayerOwnership,
    ) -> Dict[int, LayerOwnership]:
        """Gather ownership info from all ranks."""
        if not dist.is_initialized():
            return {self.rank: local_ownership}

        world_size = dist.get_world_size()
        local_data = {
            "rank": local_ownership.rank,
            "layer_names": local_ownership.layer_names,
            "is_placeholder": local_ownership.is_placeholder,
            "shard_range": local_ownership.shard_range,
            "shard_dim": local_ownership.shard_dim,
        }
        gathered_data = [None] * world_size
        dist.all_gather_object(gathered_data, local_data)

        return {
            d["rank"]: LayerOwnership(
                rank=d["rank"],
                layer_names=d["layer_names"],
                is_placeholder=d["is_placeholder"],
                shard_range=d.get("shard_range", {}),
                shard_dim=d.get("shard_dim", {}),
            )
            for d in gathered_data
            if d is not None
        }


def build_target_ownership(
    model: nn.Module,
    new_pg_mesh,
    new_encoder_plugins: dict,
    new_language_model_plugin,
    source_ownership: "Optional[Dict[int, LayerOwnership]]" = None,
) -> Dict[int, LayerOwnership]:
    """Compute target parameter ownership for the new parallel configuration.

    Each rank in the target configuration owns a specific shard of each
    parameter — determined jointly by its PP stage, DP replica, and TP rank.
    No actual tensor movement occurs; this function derives the ownership
    purely from the new ``pg_mesh`` topology and plugin pipeline templates.

    When ``source_ownership`` is supplied, any parameter that was TP-sharded
    in the source will carry a ``shard_range`` annotation in the returned
    ownership (reflecting the range the rank will hold *after* redistribution,
    i.e. the new TP shard).  This allows the executor to compute direct
    shard-to-shard transfers without any intermediate full-tensor assembly.

    Args:
        model: The model whose ``named_parameters()`` defines the universe of
            parameter names.  Only the names are used (not the values).
        new_pg_mesh: A ``MultiModalProcessGroupMesh`` for the new config.
        new_encoder_plugins: Dict of ``{name: ModalParallelPlugin}`` for
            encoders in the new configuration.
        new_language_model_plugin: ``ModalParallelPlugin`` for the LLM.
        source_ownership: Optional full source ownership map (returned by
            ``TensorOwnershipAnalyzer.analyze()``).  Used to propagate
            ``shard_dim`` and to compute the full-parameter size for each
            TP-sharded parameter.

    Returns:
        Dictionary mapping every world rank to a ``LayerOwnership`` that
        reflects the target state.  All-gathered over the world group so
        every rank receives the full map.
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    my_rank = dist.get_rank() if dist.is_initialized() else 0

    all_param_names: List[str] = [name for name, _ in model.named_parameters()]

    # Derive shard_dim and full-size for TP-sharded parameters from the
    # source ownership so that we can compute target shard ranges without
    # needing any additional communication.
    param_shard_dim: Dict[str, Optional[int]] = {}
    param_full_size: Dict[str, Optional[int]] = {}
    if source_ownership:
        for pname in all_param_names:
            dim = None
            full_size = None
            for own in source_ownership.values():
                sd = own.shard_dim.get(pname)
                sr = own.shard_range.get(pname)
                if sd is not None:
                    dim = sd
                if sr is not None:
                    full_size = max(full_size or 0, sr[1])
            param_shard_dim[pname] = dim
            param_full_size[pname] = full_size

    def _param_stage_for_modal(
        param_names: List[str],
        modules_per_stage: List[List[str]],
    ) -> Dict[str, int]:
        result = {}
        for param_name in param_names:
            for stage_idx, stage_modules in enumerate(modules_per_stage):
                for module_prefix in stage_modules:
                    if param_name == module_prefix or param_name.startswith(
                        module_prefix + "."
                    ):
                        result[param_name] = stage_idx
                        break
                if param_name in result:
                    break
        return result

    pp_axis = new_pg_mesh.pp_axis
    dp_axis = new_pg_mesh.dp_axis
    tp_axis = new_pg_mesh.tp_axis
    sp_axis = new_pg_mesh.sp_axis

    # rank → list[(param_name, shard_range_or_None, shard_dim_or_None)]
    rank_to_owned: Dict[int, List[tuple]] = {r: [] for r in range(world_size)}

    plugins_by_template = {}
    for enc_plugin in new_encoder_plugins.values():
        plugins_by_template[enc_plugin.pipeline_template] = enc_plugin
    plugins_by_template[new_language_model_plugin.pipeline_template] = (
        new_language_model_plugin
    )

    for modal_template, mesh in new_pg_mesh.modal_meshes.items():
        # mesh shape: [pp, dp, sp, tp]
        pp_size, dp_size, sp_size, tp_size = mesh.shape
        plugin = plugins_by_template.get(modal_template)
        if plugin is None:
            continue

        param_stage = _param_stage_for_modal(
            all_param_names, modal_template.modules_per_stage
        )

        for pp_idx in range(pp_size):
            stage_params = [p for p, s in param_stage.items() if s == pp_idx]
            if not stage_params:
                continue
            for dp_idx in range(dp_size):
                # Every TP rank and SP rank now owns the parameter — but each
                # TP rank holds a different shard.  SP rank 0 is canonical.
                for tp_idx in range(tp_size):
                    owner_rank = int(mesh[pp_idx, dp_idx, 0, tp_idx])
                    for pname in stage_params:
                        dim = param_shard_dim.get(pname)
                        full_size = param_full_size.get(pname)
                        if dim is not None and full_size is not None and tp_size > 1:
                            chunk = full_size // tp_size
                            sr = (tp_idx * chunk, (tp_idx + 1) * chunk)
                        else:
                            sr = None
                        rank_to_owned[owner_rank].append((pname, sr, dim))

    # Build LayerOwnership for this rank.
    owned_set: Dict[str, tuple] = {}
    for pname, sr, dim in rank_to_owned[my_rank]:
        owned_set[pname] = (sr, dim)

    all_name_set = set(all_param_names)
    local_ownership = LayerOwnership(
        rank=my_rank,
        layer_names=list(owned_set.keys()),
        is_placeholder={n: (n not in owned_set) for n in all_name_set},
        shard_range={n: owned_set[n][0] if n in owned_set else None for n in all_name_set},
        shard_dim={n: owned_set[n][1] if n in owned_set else None for n in all_name_set},
    )

    if not dist.is_initialized():
        return {my_rank: local_ownership}

    gathered = [None] * world_size
    dist.all_gather_object(gathered, {
        "rank": local_ownership.rank,
        "layer_names": local_ownership.layer_names,
        "is_placeholder": local_ownership.is_placeholder,
        "shard_range": local_ownership.shard_range,
        "shard_dim": local_ownership.shard_dim,
    })

    return {
        d["rank"]: LayerOwnership(
            rank=d["rank"],
            layer_names=d["layer_names"],
            is_placeholder=d["is_placeholder"],
            shard_range=d.get("shard_range", {}),
            shard_dim=d.get("shard_dim", {}),
        )
        for d in gathered
        if d is not None
    }
