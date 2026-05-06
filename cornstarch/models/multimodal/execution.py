from __future__ import annotations

"""Execution graph primitives for user-composed Cornstarch multimodal models.

Cornstarch multimodal composition is intentionally not represented as a single
root ``nn.Module``. Users already own the concrete language, vision, audio, and
projector modules they want to glue together, and different experiments may
route those modules in different ways. This module provides the small execution
graph layer that turns those independently-owned modules into one runnable
multimodal computation.

The public API is based on futures. Calling methods such as
``plan.run_modality_encoder(...)`` and ``plan.run_language_model(...)`` does not
execute anything immediately. Instead, each call records one node in a DAG and
returns an :class:`ExecutionFuture` that names the value produced by that node.
Passing a future into another plan method records a data dependency. At execution
time the plan topologically sorts only the nodes needed by the requested future,
runs them in dependency order, and returns the requested output.

This design keeps the graph explicit and inspectable while avoiding hidden
processor/model coupling. Users preprocess/tokenize inputs outside Cornstarch,
pass concrete tensors and modules into the plan, and can inspect the resulting
graph with ``describe()``, ``to_mermaid()``, or ``to_dot()`` before running it.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class ExecutionFuture:
    """A symbolic handle to a value that will be produced by a plan node.

    A future is returned whenever a node is added to a
    :class:`CornstarchExecutionPlan`. It is deliberately lightweight: the future
    stores the output name and a private pointer to the owning plan, but it does
    not store tensor data itself. The data only exists while executing the plan.

    Futures are how dependencies are expressed. For example, passing the future
    returned by ``run_modality_encoder`` into
    ``merge_modality_encoder_outputs`` tells the plan that the merge node cannot
    run until the modality encoder node has produced its output. Calling
    ``future.execute()`` executes only the dependency closure needed for that
    future, so intermediate futures can be run for debugging without executing
    downstream language-model nodes.
    """

    name: str
    _plan: CornstarchExecutionPlan | None = field(default=None, repr=False, compare=False)

    def execute(self, inputs: Mapping[str, Any] | None = None) -> Any:
        """Execute only this future's dependency subgraph and return its value."""
        if self._plan is None:
            raise ValueError("Cannot execute a future that is not attached to a plan.")
        return self._plan._execute_until(self.name, inputs)


@dataclass(frozen=True)
class ExecutionNode:
    """One recorded operation in the multimodal execution DAG.

    Nodes are internal records, not user-facing module wrappers. The ``kind``
    field selects the executor path, while ``params`` stores the concrete module
    instance and arguments for that operation. Arguments may contain plain values
    such as tensors, nested containers, or :class:`ExecutionFuture` instances.

    Dependencies are derived by recursively scanning ``params`` for futures.
    This keeps the plan API natural: a user passes the future object where the
    produced data should flow, and the graph dependency is inferred from that
    object rather than from a separate edge-registration API.
    """

    name: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def dependencies(self) -> set[str]:
        """Return named tensor dependencies consumed by this node."""
        return _collect_future_names(self.params)


class CornstarchExecutionPlan:
    """A user-visible execution DAG for multimodal module composition.

    The plan records calls to concrete modules without executing them. It is
    intended to be the primary orchestration object for new Cornstarch
    multimodal models: users create or load unimodal modules, wrap modality
    encoders with their projectors, then describe how those modules should be
    connected.

    The plan currently understands three high-level operations:

    - ``run_modality_encoder`` executes a concrete modality module, typically a
      ``CornstarchModalityEncoder`` that already includes encoder and projector.
    - ``merge_modality_encoder_outputs`` embeds text tokens with the language
      model embedding layer and scatters projected modality features into
      placeholder token positions.
    - ``run_language_model`` runs the language model on the merged embeddings.

    Execution is dependency driven. Insertion order is only used as a stable
    ordering when independent nodes have no dependency relation. When a future is
    executed, the plan topologically sorts the dependency closure for that
    future, runs each needed node exactly once for that execution call, and
    returns the future's concrete result. ``plan.execute()`` is a convenience for
    executing the whole graph and returning the output of the last topologically
    executed node.
    """

    def __init__(self) -> None:
        self._nodes: list[ExecutionNode] = []

    @property
    def nodes(self) -> tuple[ExecutionNode, ...]:
        """Return plan nodes in insertion order."""
        return tuple(self._nodes)

    def run_modality_encoder(
        self,
        module: Any,
        name: str | None = None,
        **inputs: Any,
    ) -> ExecutionFuture:
        """Add a modality encoder node and return its future output handle."""
        modality_name = getattr(module, "modality", None)
        default_name = (
            f"{modality_name}_encoder_outputs"
            if modality_name is not None
            else "modality_encoder_outputs"
        )
        return self._add_node(
            ExecutionNode(
                name=name or default_name,
                kind="run_modality_encoder",
                params={"module": module, "inputs": dict(inputs)},
            )
        )

    def merge_modality_encoder_outputs(
        self,
        language_model: Any,
        input_ids: Any,
        labels: Any,
        modality_token_ids: Mapping[str, int],
        encoder_outputs: Mapping[str, Any] | None = None,
        name: str | None = None,
    ) -> ExecutionFuture:
        """Add a text/modality merge node and return its future output handle."""
        return self._add_node(
            ExecutionNode(
                name=name or "merged_language_inputs",
                kind="merge_modality_encoder_outputs",
                params={
                    "language_model": language_model,
                    "input_ids": input_ids,
                    "labels": labels,
                    "encoder_outputs": dict(encoder_outputs or {}),
                    "modality_token_ids": dict(modality_token_ids),
                },
            )
        )

    def run_language_model(
        self,
        module: Any,
        inputs: Any = ExecutionFuture("merged_language_inputs"),
        name: str | None = None,
    ) -> ExecutionFuture:
        """Add a language model execution node and return its future output handle."""
        return self._add_node(
            ExecutionNode(
                name=name or "language_outputs",
                kind="run_language_model",
                params={"module": module, "inputs": inputs},
            )
        )

    def validate(self) -> None:
        """Validate graph shape and model references."""
        self._topological_nodes()

    def execute(
        self,
        inputs: Mapping[str, Any] | None = None,
    ) -> Any:
        """Execute the DAG and return the final node output."""
        ordered_nodes = self._topological_nodes()
        if not ordered_nodes:
            raise ValueError("Execution plan has no nodes to execute.")
        return self._execute_nodes(ordered_nodes, inputs)

    def describe(self) -> str:
        """Return a compact text description of the execution DAG."""
        lines = ["CornstarchExecutionPlan:"]
        for node in self._nodes:
            dependencies = ", ".join(sorted(node.dependencies)) or "none"
            lines.append(f"- {node.name}: {node.kind} <- {dependencies}")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Render the execution DAG as a Mermaid flowchart."""
        lines = ["flowchart LR"]
        produced = {node.name for node in self._nodes}
        for node in self._nodes:
            node_id = self._graph_id(node.name)
            lines.append(f'    {node_id}["{node.name}: {node.kind}"]')
            for dependency in sorted(node.dependencies):
                dep_id = self._graph_id(dependency)
                if dependency not in produced:
                    lines.append(f'    {dep_id}["input: {dependency}"]')
                lines.append(f"    {dep_id} --> {node_id}")
        return "\n".join(lines)

    def to_dot(self) -> str:
        """Render the execution DAG in Graphviz DOT format."""
        lines = ["digraph CornstarchExecutionPlan {"]
        produced = {node.name for node in self._nodes}
        for node in self._nodes:
            node_id = self._graph_id(node.name)
            lines.append(f'  {node_id} [label="{node.name}: {node.kind}"];')
            for dependency in sorted(node.dependencies):
                dep_id = self._graph_id(dependency)
                if dependency not in produced:
                    lines.append(f'  {dep_id} [label="input: {dependency}"];')
                lines.append(f"  {dep_id} -> {node_id};")
        lines.append("}")
        return "\n".join(lines)

    def _add_node(self, node: ExecutionNode) -> ExecutionFuture:
        if any(existing.name == node.name for existing in self._nodes):
            raise ValueError(f"Duplicate execution output name: {node.name}")
        self._nodes.append(node)
        return ExecutionFuture(node.name, self)

    def _topological_nodes(self, target_name: str | None = None) -> list[ExecutionNode]:
        nodes_by_name = {node.name: node for node in self._nodes}
        if len(nodes_by_name) != len(self._nodes):
            raise ValueError("Execution plan contains duplicate node names.")
        if target_name is not None and target_name not in nodes_by_name:
            raise ValueError(f"Execution plan does not contain a node named {target_name!r}.")

        dependencies_by_name = {
            node.name: {dep for dep in node.dependencies if dep in nodes_by_name}
            for node in self._nodes
        }
        ordered: list[ExecutionNode] = []
        temporary: set[str] = set()
        permanent: set[str] = set()

        def visit(name: str) -> None:
            if name in permanent:
                return
            if name in temporary:
                raise ValueError("Execution plan contains a cycle.")
            temporary.add(name)
            for dependency in sorted(dependencies_by_name[name]):
                visit(dependency)
            temporary.remove(name)
            permanent.add(name)
            ordered.append(nodes_by_name[name])

        if target_name is None:
            for node in self._nodes:
                visit(node.name)
        else:
            visit(target_name)
        return ordered

    def _execute_until(
        self,
        target_name: str,
        inputs: Mapping[str, Any] | None = None,
    ) -> Any:
        """Execute only the dependency closure needed for one named output."""
        return self._execute_nodes(self._topological_nodes(target_name), inputs)

    def _execute_nodes(
        self,
        ordered_nodes: list[ExecutionNode],
        inputs: Mapping[str, Any] | None = None,
    ) -> Any:
        values = dict(inputs or {})
        last_output = None
        for node in ordered_nodes:
            missing = sorted(dep for dep in node.dependencies if dep not in values)
            if missing:
                raise ValueError(
                    f"Node {node.name!r} cannot run because inputs are missing: {missing}"
                )
            last_output = self._execute_node(node, values)
            values[node.name] = last_output
        return last_output

    @staticmethod
    def _execute_node(node: ExecutionNode, values: Mapping[str, Any]) -> Any:
        if node.kind == "run_modality_encoder":
            encoder_inputs = {
                arg_name: _resolve_value(value, values)
                for arg_name, value in node.params["inputs"].items()
            }
            return node.params["module"](**encoder_inputs)
        if node.kind == "merge_modality_encoder_outputs":
            return merge_modality_encoder_outputs(
                language_model=node.params["language_model"],
                input_ids=_resolve_value(node.params["input_ids"], values),
                labels=_resolve_value(node.params["labels"], values),
                encoder_outputs={
                    modality: _resolve_value(value, values)
                    for modality, value in node.params["encoder_outputs"].items()
                },
                modality_token_ids=node.params["modality_token_ids"],
            )
        if node.kind == "run_language_model":
            return node.params["module"](
                input_ids=None,
                **dict(_resolve_value(node.params["inputs"], values)),
            )
        raise ValueError(f"Unsupported execution node kind: {node.kind}")

    @staticmethod
    def _graph_id(name: str) -> str:
        return "".join(char if char.isalnum() else "_" for char in name)


def _collect_future_names(value: Any) -> set[str]:
    if isinstance(value, ExecutionFuture):
        return {value.name}
    if isinstance(value, Mapping):
        names: set[str] = set()
        for item in value.values():
            names.update(_collect_future_names(item))
        return names
    if isinstance(value, (list, tuple)):
        names: set[str] = set()
        for item in value:
            names.update(_collect_future_names(item))
        return names
    return set()


def _resolve_value(value: Any, values: Mapping[str, Any]) -> Any:
    if isinstance(value, ExecutionFuture):
        return values[value.name]
    if isinstance(value, Mapping):
        return {
            key: _resolve_value(item, values)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_resolve_value(item, values) for item in value)
    if isinstance(value, list):
        return [_resolve_value(item, values) for item in value]
    return value


def merge_modality_encoder_outputs(
    language_model: Any,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    encoder_outputs: Mapping[str, Any],
    modality_token_ids: Mapping[str, int],
) -> dict[str, torch.Tensor]:
    """Build language-model inputs by replacing modality placeholder tokens.

    ``input_ids`` is expected to already contain one special token position for
    every projected modality feature. ``modality_token_ids`` maps modality names
    such as ``"vision"`` or ``"audio"`` to those special token IDs. The merge
    operation masks those special token IDs to a safe ordinary token before
    calling the language model's embedding layer, then scatters projected
    modality features into the matching embedding positions.

    The operation validates that the number of placeholder tokens for each
    modality exactly matches the number of feature rows supplied for that
    modality. Labels at modality positions are masked to ``-100`` because those
    positions are inputs to the language model, not language targets. The
    returned dictionary is shaped for ``run_language_model``:
    ``inputs_embeds``, a generated boolean ``attention_mask``, and masked
    ``labels``.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must be a 2D tensor of shape (batch, sequence).")
    if labels.shape != input_ids.shape:
        raise ValueError("labels must have the same shape as input_ids.")

    token_ids = {
        modality: int(token_id)
        for modality, token_id in modality_token_ids.items()
    }
    token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in token_ids.values():
        token_mask |= input_ids == token_id

    safe_input_ids = input_ids.masked_fill(token_mask, 0)
    inputs_embeds = language_model.pre_decoder["embed_tokens"](safe_input_ids)
    labels = labels.masked_fill(token_mask, -100)

    for modality, output in encoder_outputs.items():
        if modality not in token_ids:
            raise ValueError(f"Token ID for modality {modality!r} was not provided.")
        features = _first_output_tensor(output)
        if features.ndim > 2:
            features = features.reshape(-1, features.shape[-1])
        if features.shape[-1] != inputs_embeds.shape[-1]:
            raise ValueError(
                f"Expected {modality} features hidden size {features.shape[-1]} "
                f"to match language hidden size {inputs_embeds.shape[-1]}."
            )

        modality_mask = input_ids == token_ids[modality]
        num_tokens = int(modality_mask.sum().item())
        if num_tokens != features.shape[0]:
            raise ValueError(
                f"Number of {modality} tokens {num_tokens} must equal number of "
                f"{modality} features {features.shape[0]}."
            )
        expanded_mask = modality_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            expanded_mask.to(inputs_embeds.device),
            features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
        )

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.bool,
            device=inputs_embeds.device,
        ),
        "labels": labels,
    }


def _first_output_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, Mapping):
        return output["last_hidden_state"]
    if isinstance(output, tuple):
        return output[0]
    raise TypeError(f"Cannot extract hidden states from output of type {type(output).__name__}.")
