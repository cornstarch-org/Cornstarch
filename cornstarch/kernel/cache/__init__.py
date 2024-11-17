from .cache_manager import CACHE_PATH, get_cached_kernels
import os
import shutil
import glob

__all__ = ["CACHE_PATH", "get_cached_kernels"]

# first try to use cache_inductor, store them in $HOME/.cache_inductor
# if there are already some cache files in the directory, remove them

# Only clean if cache directory exists and has files
if os.path.exists(CACHE_PATH):
    files = glob.glob(os.path.join(CACHE_PATH, "*"))
    if files:
        marker_file = os.path.join(CACHE_PATH, ".cleanup_marker")
        try:
            # os.O_EXCL ensures the file must not exist
            fd = os.open(marker_file, os.O_CREAT | os.O_EXCL)
            os.close(fd)
            # Only the process that successfully created the marker will clean
            # print(f"remove cache files in {CACHE_PATH}")
            shutil.rmtree(CACHE_PATH)
            os.makedirs(CACHE_PATH)
        except FileExistsError:
            # Another process is handling the cleanup
            pass
else:
    os.makedirs(CACHE_PATH)


os.environ["TORCHINDUCTOR_CACHE_DIR"] = (
    CACHE_PATH  # https://github.com/pytorch/pytorch/issues/121122
)
