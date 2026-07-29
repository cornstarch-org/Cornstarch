"""``apply_tensor_parallel`` raises an explicit error for unregistered families.

The earlier trial silently no-op'd when a model family had no TP plan, which
let callers believe a model was tensor-parallel when it was not.  The plan
lookup now happens before any mesh use, so this is a plain (non-distributed)
unit test: building a model whose config class is not in the TP registry and
calling ``apply_tensor_parallel`` must raise ``TensorParallelNotSupportedError``.
"""

import unittest

from tests.model.model_configs import nemotron_h_config

from cornstarch.distributed.tensor_parallel import (
    TensorParallelNotSupportedError,
    apply_tensor_parallel,
)
from cornstarch.distributed.tensor_parallel.plans import get_tp_plan
from cornstarch.models import from_hf_config


class TestUnregisteredTensorParallel(unittest.TestCase):
    def test_unregistered_family_raises(self):
        config = nemotron_h_config()
        self.assertIsNone(
            get_tp_plan(type(config).__name__),
            "test requires an unregistered config family",
        )
        model = from_hf_config(config, model_kind="language")

        # The plan lookup raises before the mesh is ever touched, so a real
        # process group / DeviceMesh is unnecessary here.
        with self.assertRaises(TensorParallelNotSupportedError):
            apply_tensor_parallel(model, tp_mesh=None)

    def test_registered_family_has_plan(self):
        from tests.model.model_configs import llama_config

        self.assertIsNotNone(get_tp_plan(type(llama_config()).__name__))

    def test_requested_dense_tp_plans_resolve_every_path(self):
        from tests.model.model_configs import (
            clip_vision_config,
            gemma4_vision_config,
            llama4_config,
            siglip2_vision_config,
            whisper_config,
        )

        cases = (
            (clip_vision_config(), "vision"),
            (gemma4_vision_config(), "vision"),
            (llama4_config(), "language"),
            (siglip2_vision_config(), "vision"),
            (whisper_config(), "audio"),
        )
        for config, model_kind in cases:
            plan = get_tp_plan(type(config).__name__)
            self.assertIsNotNone(plan)
            model = from_hf_config(config, model_kind=model_kind)
            repeated = getattr(model, model._section_names()[1])
            for layer in repeated:
                for path in plan:
                    self.assertIsNotNone(layer.get_submodule(path))


if __name__ == "__main__":
    unittest.main()
