import contextlib
import os
import sys

sys.path.append("/usr/local/dcgm/bindings/python3")
import dcgm_fields
import dcgm_structs
import pydcgm
from DcgmReader import DcgmReader


class DcgmContextManager:
    def __init__(self):
        gpu_id = int(os.environ["LOCAL_RANK"])
        reader = DcgmReader(
            fieldIds=[
                dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY,
            ],
            updateFrequency=10,
            # GPU indices based on ordering on the host (where DCGM daemon is running).
            # This ignores CUDA_VISIBLE_DEVICES and whatever's mounted in Docker.
            gpuIds=[gpu_id],
        )

        reader.m_dcgmHandle = pydcgm.DcgmHandle(
            opMode=dcgm_structs.DCGM_OPERATION_MODE_AUTO
        )

        reader.InitializeFromHandle()

        self.reader = reader
        self.gpu_id = gpu_id
        self.data: dict[
            int, dict[str, list[pydcgm.dcgm_field_helpers.DcgmFieldValue]]
        ] = None

    @contextlib.contextmanager
    def profile(self):
        # Discard the previous result
        _ = self.reader.GetAllGpuValuesAsFieldNameDictSinceLastCall()

        yield

        # Store the result of the second call
        self.data = self.reader.GetAllGpuValuesAsFieldNameDictSinceLastCall()

    # list of (timestamp, sm_utilization value)
    def get_sm_occupancy_trace(self) -> list[tuple[int, float]]:
        if self.data is None:
            raise RuntimeError("Must call profile() first.")

        assert len(self.data) == 1, "Only supports one GPU."
        data: list[pydcgm.dcgm_field_helpers.DcgmFieldValue] = self.data[self.gpu_id][
            "sm_occupancy"
        ]

        return [(field.ts, field.value) for field in data]
