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
        """
        This requires the DCGM daemon to be running on the host.
        Use:
        $ systemctl --now enable nvidia-dcgm
        or manually spawn a process with:
        $ nv-hostengine
        """
        gpu_id = int(os.environ["LOCAL_RANK"])
        reader = DcgmReader(
            fieldIds=[
                dcgm_fields.DCGM_FI_PROF_SM_OCCUPANCY,
                dcgm_fields.DCGM_FI_PROF_PIPE_TENSOR_ACTIVE,  # tensor core utilization
                dcgm_fields.DCGM_FI_PROF_SM_ACTIVE,
                dcgm_fields.DCGM_FI_DEV_FB_USED,  # memory usage
            ],
            updateFrequency=1000,  # 1ms
            fieldGroupName=str(os.getpid()),
            # GPU indices based on ordering on the host (where DCGM daemon is running).
            # This ignores CUDA_VISIBLE_DEVICES and whatever's mounted in Docker.
            gpuIds=[gpu_id],
        )

        reader.m_dcgmHandle = pydcgm.DcgmHandle(
            opMode=dcgm_structs.DCGM_OPERATION_MODE_AUTO,
            ipAddress="localhost:5555",
        )

        reader.InitializeFromHandle()
        # Flush
        _ = reader.GetAllGpuValuesAsFieldNameDictSinceLastCall()

        self.reader = reader
        self.gpu_id = gpu_id
        self.data: dict[
            int, dict[str, list[pydcgm.dcgm_field_helpers.DcgmFieldValue]]
        ] = None

    def shutdown(self):
        self.reader.Shutdown()

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

    def get_sm_activity_trace(self) -> list[tuple[int, float]]:
        if self.data is None:
            raise RuntimeError("Must call profile() first.")

        assert len(self.data) == 1, "Only supports one GPU."
        data: list[pydcgm.dcgm_field_helpers.DcgmFieldValue] = self.data[self.gpu_id][
            "sm_active"
        ]

        return [(field.ts, field.value) for field in data]

    def get_tensor_core_util_trace(self) -> list[tuple[int, float]]:
        if self.data is None:
            raise RuntimeError("Must call profile() first.")

        assert len(self.data) == 1, "Only supports one GPU."
        data: list[pydcgm.dcgm_field_helpers.DcgmFieldValue] = self.data[self.gpu_id][
            "tensor_active"
        ]

        return [(field.ts, field.value) for field in data]

    def get_memory_trace(self) -> list[tuple[int, float]]:
        if self.data is None:
            raise RuntimeError("Must call profile() first.")

        assert len(self.data) == 1, "Only supports one GPU."
        data: list[pydcgm.dcgm_field_helpers.DcgmFieldValue] = self.data[self.gpu_id][
            "fb_used"
        ]

        return [(field.ts, field.value) for field in data]
