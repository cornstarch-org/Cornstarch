from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, Iterable, Optional, Union

import torch
from colossalai.accelerator import get_accelerator
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.pipeline.schedule._utils import merge_batch, retain_grad, tree_map
from colossalai.pipeline.weight_grad_store import WeightGradStore
from torch import nn

from cornstarch.plugin.multimodal_parallel_plugin.multimodal_1f1b import (
    MultimodalEncoderTrainingOneForwardOneBackwardSchedule,
)
from cornstarch.plugin.multimodal_parallel_plugin.multimodal_stage_manager import (
    MultiModalPipelineStageManager,
)


class MultimodalEncoderTrainingZeroBubblePipelineSchedule(
    MultimodalEncoderTrainingOneForwardOneBackwardSchedule
):
    """Multimodal zero-bubble pipeline schedule (single chunk per stage).

    This schedule keeps the multimodal border-map communication behavior from
    `MultimodalEncoderTrainingOneForwardOneBackwardSchedule`, while changing the
    execution order to ZBPP and splitting backward into:
    - backward_b_step (dgrad): compute gradients for stage inputs.
    - backward_w_step (wgrad): materialize parameter gradients via WeightGradStore.
    """

    stage_manager: MultiModalPipelineStageManager

    def _get_global_stage_info(self) -> tuple[int, int]:
        """Return `(i, p)` where `i` is global stage index and `p` total PP stages."""
        pg_mesh = self.stage_manager.pg_mesh
        assert pg_mesh.llm_template is not None, "LLM template must exist for multimodal ZBPP."
        assert len(pg_mesh.encoder_templates) == 1, (
            "Multimodal ZBPP currently supports exactly one encoder template."
        )

        encoder_template = next(iter(pg_mesh.encoder_templates.keys()))
        llm_template = pg_mesh.llm_template[0]
        p = encoder_template.num_stages + llm_template.num_stages

        my_modal = pg_mesh.my_modal
        if my_modal == encoder_template:
            i = self.stage_manager.stage_in_modal
        elif my_modal == llm_template:
            i = encoder_template.num_stages + self.stage_manager.stage_in_modal
        else:
            raise RuntimeError(f"Unsupported modal for ZBPP schedule: {my_modal}")

        return i, p

    def _no_sync_context(self, model: nn.Module, optimizer: OptimizerWrapper):
        try:
            return optimizer.no_sync()
        except AttributeError:
            return model.no_sync() if hasattr(model, "no_sync") else nullcontext()

    def backward_b_step(
        self,
        model: nn.Module,
        optimizer: OptimizerWrapper,
        input_obj: Optional[dict],
        output_obj: Union[dict, torch.Tensor],
        output_obj_grad: Optional[dict],
    ) -> Optional[dict]:
        """Backward dgrad step: compute and return gradient for `input_obj`."""
        tree_map(retain_grad, input_obj)

        with self._no_sync_context(model, optimizer):
            if output_obj_grad is None:
                optimizer.backward(output_obj, retain_graph=False)
            else:
                keys = output_obj.get("backward_tensor_keys", output_obj_grad.keys())
                tensors_to_backward = []
                grads_to_backward = []
                for k in keys:
                    if isinstance(output_obj[k], torch.Tensor):
                        tensors_to_backward.append(output_obj[k])
                        grads_to_backward.append(output_obj_grad[k])
                if len(tensors_to_backward) == 1:
                    optimizer.backward_by_grad(
                        tensors_to_backward[0],
                        grads_to_backward[0],
                        retain_graph=False,
                    )
                else:
                    optimizer.backward_by_grad(
                        tensors_to_backward,
                        grads_to_backward,
                        retain_graph=False,
                    )

        input_obj_grad = None
        if input_obj is not None:
            input_obj_grad = {}
            for k, v in input_obj.items():
                if isinstance(v, torch.Tensor) and v.grad is not None:
                    input_obj_grad[k] = v.grad
                elif isinstance(v, list):
                    input_obj_grad[k] = [item.grad for item in v]
        return input_obj_grad

    def backward_w_step(self) -> None:
        """Backward wgrad step: materialize parameter grads from WeightGradStore."""
        WeightGradStore.pop(chunk=0)

    def run_forward_backward(
        self,
        model: nn.Module,
        data_iter: Iterable,
        criterion: Callable[..., Any],
        optimizer: OptimizerWrapper | None = None,
        return_loss: bool = False,
        return_outputs: bool = False,
    ) -> dict:
        assert not self.forward_only
        assert optimizer is not None, "Optimizer is required when running backward."

        i, p = self._get_global_stage_info()
        if self.num_microbatches < 2 * p:
            raise ValueError(
                f"Zero-bubble pipeline schedule requires num_microbatches >= {2 * p}, "
                f"but got {self.num_microbatches}."
            )

        self.load_batch(data_iter)

        input_objs: list[Any] = []
        output_objs: list[Any] = []

        accum_loss = None
        if return_loss and self.stage_manager.is_last_stage(check_only_in_modal=False):
            accum_loss = torch.scalar_tensor(
                0, device=get_accelerator().get_current_device()
            )
        outputs = (
            []
            if return_outputs
            and self.stage_manager.is_last_stage(check_only_in_modal=False)
            else None
        )

        f_scheduled = 0
        bi_scheduled = 0
        bp_scheduled = 0

        warmup1_target = (p - i - 1) * 2
        warmup2_target = (p - 1) * 2

        # Phase 1: warmup1 (F only)
        while f_scheduled < warmup1_target:
            input_obj = self.recv_forward()
            output_obj = self.forward_step(
                model, input_obj, criterion, accum_loss, outputs
            )
            self.send_forward(output_obj)
            input_objs.append(input_obj)
            output_objs.append(output_obj)
            f_scheduled += 1

        # Phase 2: warmup2 (F + BI)
        while f_scheduled < warmup2_target:
            input_obj = self.recv_forward()
            output_obj = self.forward_step(
                model, input_obj, criterion, accum_loss, outputs
            )
            self.send_forward(output_obj)
            input_objs.append(input_obj)
            output_objs.append(output_obj)
            f_scheduled += 1

            bi_input_obj = input_objs.pop(0)
            bi_output_obj = output_objs.pop(0)
            output_obj_grad = self.recv_backward()
            input_obj_grad = self.backward_b_step(
                model, optimizer, bi_input_obj, bi_output_obj, output_obj_grad
            )
            self.send_backward(bi_input_obj, input_obj_grad)
            WeightGradStore.flush(chunk=0)
            bi_scheduled += 1

        # Phase 3: steady (F + BI + BP)
        while f_scheduled < self.num_microbatches:
            input_obj = self.recv_forward()
            output_obj = self.forward_step(
                model, input_obj, criterion, accum_loss, outputs
            )
            self.send_forward(output_obj)
            input_objs.append(input_obj)
            output_objs.append(output_obj)
            f_scheduled += 1

            bi_input_obj = input_objs.pop(0)
            bi_output_obj = output_objs.pop(0)
            output_obj_grad = self.recv_backward()
            input_obj_grad = self.backward_b_step(
                model, optimizer, bi_input_obj, bi_output_obj, output_obj_grad
            )
            self.send_backward(bi_input_obj, input_obj_grad)
            WeightGradStore.flush(chunk=0)
            bi_scheduled += 1

            self.backward_w_step()
            bp_scheduled += 1

        # Phase 4: cooldown1 (BI + BP)
        while bi_scheduled < self.num_microbatches:
            bi_input_obj = input_objs.pop(0)
            bi_output_obj = output_objs.pop(0)
            output_obj_grad = self.recv_backward()
            input_obj_grad = self.backward_b_step(
                model, optimizer, bi_input_obj, bi_output_obj, output_obj_grad
            )
            self.send_backward(bi_input_obj, input_obj_grad)
            WeightGradStore.flush(chunk=0)
            bi_scheduled += 1

            self.backward_w_step()
            bp_scheduled += 1

        # Phase 5: cooldown2 (BP only)
        while bp_scheduled < self.num_microbatches:
            self.backward_w_step()
            bp_scheduled += 1

        assert f_scheduled == self.num_microbatches
        assert bi_scheduled == self.num_microbatches
        assert bp_scheduled == self.num_microbatches
        assert len(input_objs) == 0 and len(output_objs) == 0

        if outputs is not None:
            if isinstance(model, ModelWrapper):
                model = model.unwrap()
            batch_size_dim = getattr(model, "batch_size_dim", 0)
            outputs = merge_batch(outputs, batch_size_dim)
        return {"loss": accum_loss, "outputs": outputs}

