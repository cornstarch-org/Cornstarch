"""Unit tests for the count-heuristic ZB-H2 program and split B/W."""
from __future__ import annotations

import copy
import inspect
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from cornstarch.distributed.parallel_config import ParallelConfig
from cornstarch.distributed.parallelization import ParallelizationPlan
from cornstarch.distributed.pipeline_parallel.deferred_weight_grad import (
    DeferredWeightGradStore,
    apply_deferred_weight_gradients,
    deferred_expert_linear,
)
from cornstarch.distributed.pipeline_parallel.schedule_zbpp import (
    OperationType,
    ZeroBubblePipelineSchedule,
    h2_warmup_for_stage,
    select_main_operations,
)


class TestZBH2Program(unittest.TestCase):
    def test_h2_counts_for_two_three_and_four_stages(self):
        for num_stages in (2, 3, 4):
            num_microbatches = 2 * num_stages
            warmups = [
                h2_warmup_for_stage(num_stages, stage)
                for stage in range(num_stages)
            ]
            self.assertTrue(
                all(left - right == 2 for left, right in zip(warmups, warmups[1:]))
            )

            for num_warmup in warmups:
                f, b, w = num_warmup, 0, 0
                programs = []
                counts = {
                    OperationType.FORWARD: num_warmup,
                    OperationType.BACKWARD: 0,
                    OperationType.WEIGHT: 0,
                }
                while f < num_microbatches or b < num_microbatches or w < num_microbatches:
                    operations = select_main_operations(f, b, w, num_microbatches)
                    programs.append("".join(operation.type.value for operation in operations))
                    for operation in operations:
                        counts[operation.type] += 1
                        if operation.type is OperationType.FORWARD:
                            f += 1
                        elif operation.type is OperationType.BACKWARD:
                            b += 1
                        else:
                            w += 1
                    self.assertLessEqual(w, b)
                    self.assertLessEqual(b, f)
                    self.assertLessEqual(f, num_microbatches)

                self.assertEqual(programs.count("FBW"), num_microbatches - num_warmup)
                self.assertEqual(programs.count("BW"), num_warmup)
                self.assertNotIn("W", programs)
                self.assertEqual(set(counts.values()), {num_microbatches})

    def test_operation_checks_are_independent_and_count_only(self):
        signature = inspect.signature(select_main_operations)
        self.assertEqual(
            list(signature.parameters),
            ["forward_count", "backward_count", "weight_count", "num_microbatches"],
        )
        self.assertEqual(
            [op.type.value for op in select_main_operations(3, 1, 1, 4)],
            ["F", "B", "W"],
        )
        self.assertEqual(
            [op.type.value for op in select_main_operations(4, 2, 2, 4)],
            ["B", "W"],
        )
        self.assertEqual(
            [op.type.value for op in select_main_operations(4, 4, 3, 4)],
            ["W"],
        )

    def test_rejects_too_few_microbatches(self):
        schedule = ZeroBubblePipelineSchedule.__new__(ZeroBubblePipelineSchedule)
        schedule._idle = False
        schedule._num_stages = 2
        with self.assertRaisesRegex(ValueError, "at least 2p-1"):
            schedule.step([{}, {}], lambda *_: None)


class TestDeferredWeightGradient(unittest.TestCase):
    def test_dense_bias_and_microbatch_accumulation_match_backward(self):
        torch.manual_seed(7)
        baseline = nn.Sequential(
            nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 3, bias=True)
        )
        split = copy.deepcopy(baseline)
        apply_deferred_weight_gradients(split)
        store = DeferredWeightGradStore()

        baseline_inputs = [torch.randn(3, 4, requires_grad=True) for _ in range(3)]
        split_inputs = [value.detach().clone().requires_grad_(True) for value in baseline_inputs]
        for value in baseline_inputs:
            baseline(value).square().mean().backward()

        for microbatch, value in enumerate(split_inputs):
            with store.capture(microbatch):
                split(value).square().mean().backward()
            if microbatch == 0:
                weight_grads = [
                    parameter.grad
                    for name, parameter in split.named_parameters()
                    if name.endswith("weight")
                ]
                self.assertTrue(all(grad is None for grad in weight_grads))
            store.execute(microbatch)

        for baseline_parameter, split_parameter in zip(
            baseline.parameters(), split.parameters()
        ):
            self.assertTrue(
                torch.allclose(
                    baseline_parameter.grad, split_parameter.grad, atol=1e-6
                )
            )
        for baseline_input, split_input in zip(baseline_inputs, split_inputs):
            self.assertTrue(torch.allclose(baseline_input.grad, split_input.grad, atol=1e-6))
        store.assert_empty()

    def test_tied_weight_accumulates_embedding_b_and_linear_w(self):
        class TiedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(11, 5)
                self.output = nn.Linear(5, 11, bias=False)
                self.output.weight = self.embedding.weight

            def forward(self, tokens):
                hidden = self.embedding(tokens).mean(dim=1)
                return self.output(hidden)

        torch.manual_seed(11)
        baseline = TiedModel()
        split = copy.deepcopy(baseline)
        apply_deferred_weight_gradients(split)
        tokens = torch.randint(0, 11, (4, 3))
        baseline(tokens).square().sum().backward()

        store = DeferredWeightGradStore()
        with store.capture(0):
            split(tokens).square().sum().backward()
        # The tied embedding contribution is intentionally produced during B.
        self.assertIsNotNone(split.embedding.weight.grad)
        store.execute(0)
        self.assertTrue(
            torch.allclose(
                baseline.embedding.weight.grad,
                split.embedding.weight.grad,
                atol=1e-6,
            )
        )

    def test_stacked_expert_weight_gradient_matches_linear(self):
        torch.manual_seed(19)
        inputs = torch.randn(5, 4, requires_grad=True)
        split_inputs = inputs.detach().clone().requires_grad_(True)
        weights = nn.Parameter(torch.randn(3, 6, 4))
        split_weights = nn.Parameter(weights.detach().clone())
        F.linear(inputs, weights[1]).square().sum().backward()

        store = DeferredWeightGradStore()
        with store.capture(0):
            deferred_expert_linear(split_inputs, split_weights, 1).square().sum().backward()
        self.assertIsNone(split_weights.grad)
        store.execute(0)
        self.assertTrue(torch.allclose(weights.grad, split_weights.grad, atol=1e-6))
        self.assertTrue(torch.allclose(inputs.grad, split_inputs.grad, atol=1e-6))


class TestZBPPConfiguration(unittest.TestCase):
    def test_invalid_schedule_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "pipeline_schedule"):
            ParallelConfig(pipeline_schedule="vpp")

    def test_pipelined_modules_must_select_one_schedule(self):
        plan = ParallelizationPlan()
        plan._modules = [object(), object()]
        plan._configs = [
            ParallelConfig(pipeline_parallel_size=1, pipeline_schedule="1f1b"),
            ParallelConfig(pipeline_parallel_size=1, pipeline_schedule="zbpp"),
        ]
        with self.assertRaisesRegex(ValueError, "same pipeline_schedule"):
            plan._assign_ranks([0, 1], 2)


if __name__ == "__main__":
    unittest.main()
