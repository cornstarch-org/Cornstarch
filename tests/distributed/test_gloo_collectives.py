"""Regression tests for the CPU/Gloo collective shims used by distributed tests."""

import torch
import torch.distributed as dist

from tests.distributed.distributed_base import GlooDistributedTestBase


class TestVariableAllToAllSingleGloo(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def test_routes_to_clockwise_peer_and_receives_from_counterclockwise_peer(self):
        """A ring step must not use the receive peer as its send destination."""
        rank = dist.get_rank()
        send_counts = (
            (1, 2, 0, 1),
            (0, 1, 2, 0),
            (1, 0, 1, 2),
            (2, 1, 0, 1),
        )
        input_chunks = [
            torch.tensor(
                [100 * rank + 10 * destination + index for index in range(count)],
                dtype=torch.long,
            )
            for destination, count in enumerate(send_counts[rank])
        ]
        input_tensor = torch.cat(input_chunks)
        output_counts = [send_counts[source][rank] for source in range(self.world_size)]
        output = torch.empty(sum(output_counts), dtype=torch.long)

        dist.all_to_all_single(
            output,
            input_tensor,
            output_split_sizes=output_counts,
            input_split_sizes=list(send_counts[rank]),
            group=dist.group.WORLD,
        )

        expected = torch.cat(
            [
                torch.tensor(
                    [100 * source + 10 * rank + index for index in range(count)],
                    dtype=torch.long,
                )
                for source, count in enumerate(output_counts)
            ]
        )
        torch.testing.assert_close(output, expected)
