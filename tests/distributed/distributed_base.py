import os
import random
import sys
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import TEST_SKIPS, MultiProcessTestCase
from torch.testing._internal.common_utils import FILE_SCHEMA

from .gloo_utils import (
    all_gather_gloo,
    all_to_all_gloo,
    all_to_all_single_gloo,
    batch_isend_irecv_gloo,
    reduce_scatter_gloo,
)


class GlooDistributedTestBase(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        raise NotImplementedError

    def setUp(self) -> None:
        super().setUp()
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}):
            self._spawn_processes()

    def tearDown(self) -> None:
        return super().tearDown()

    @classmethod
    def _run(cls, rank: int, test_name: str, file_name: str, pipe, **kwargs) -> None:
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        print(f"dist init r={self.rank}, world={self.world_size}")

        backend = "gloo"

        try:
            dist.init_process_group(
                init_method=f"{FILE_SCHEMA}{self.file_name}",
                backend=backend,
                world_size=int(self.world_size),
                rank=self.rank,
            )
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)

            raise

        # Spawned subprocesses do not reliably inherit the pytest-set
        # TORCHDYNAMO_DISABLE env var, and Cornstarch enables per-layer
        # torch.compile by default. Force-disable dynamo so the gloo CPU tests
        # run eager (inductor cannot compile the gloo-patched collectives).
        import torch._dynamo

        torch._dynamo.config.disable = True
        os.environ["TORCHDYNAMO_DISABLE"] = "1"

        if torch.cuda.is_available() and torch.cuda.device_count():
            device_id = self.rank % torch.cuda.device_count()
            torch.cuda.set_device(device_id)

        self.reset_seed()

        dist.barrier()

        with (
            torch.backends.cudnn.flags(
                enabled=False, deterministic=True, benchmark=False
            ),
            patch.object(dist, "batch_isend_irecv", new=batch_isend_irecv_gloo),
            patch.object(dist, "all_to_all", new=all_to_all_gloo),
            patch.object(dist, "all_to_all_single", new=all_to_all_single_gloo),
            patch.object(dist, "reduce_scatter", new=reduce_scatter_gloo),
            patch.object(dist, "all_gather", new=all_gather_gloo),
        ):
            torch.use_deterministic_algorithms(mode=True)
            self.run_test(test_name, pipe)

        try:
            dist.barrier()
            dist.destroy_process_group()
        except (ValueError, RuntimeError):
            pass

    def reset_seed(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
