import contextlib
import functools
import importlib
import logging
from collections import defaultdict
from typing import Any, Callable

import torch


class TimerContextManager:
    """
    A context manager that wraps specific code and times its execution.
    It uses toch.cuda.Events to measure the time, and event objects are internally managed.
    To get elapsed time, torch.cuda.synchronize() must be called. Therefore, it is not a good idea
    to get elapsed time several times; instead, add measurement however you want,
    and get all elapsed time at once using `get_elapsed_times`.

    Example:
    ```
    manager = TimerContextManager()
    with manager.measure("torch.nn.linear.Linear.forward"):
        # Call the function you want to measure

    manager.get_elapsed_time()
    ```

    Example output of `get_elapsed_time()` (time in ms):
    [('torch.nn.Linear.forward#0', 0.06758400052785873),
     ('torch.nn.Linear.forward#1', 0.04710400104522705)]
    """

    def __init__(self):
        self.events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.original_functions: dict[str, tuple[str, str, Callable]] = {}
        self.logger = logging.getLogger(__name__)

    def wrapper(self, original_function: Callable, full_function_path: str):
        """
        A wrapper function that wraps the target function with torch.cuda.Event.
        """

        @functools.wraps(original_function)
        def wrapped_function(*args, **kwargs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            result = original_function(*args, **kwargs)
            end_event.record()

            # Save the events
            self.events.append((full_function_path, start_event, end_event))

            return result

        return wrapped_function

    @contextlib.contextmanager
    def measure(self, full_function_paths: str | list[str]):
        """
        Wrap the target function with the wrapper and measure the elapsed time.
        When it is done, restore the original function.

        For the same function, only wrap once; otherwise it raises a ValueError.
        """
        self.logger.debug(f"Measuring {full_function_paths}")

        if isinstance(full_function_paths, str):
            full_function_paths = [full_function_paths]

        for full_function_path in full_function_paths:
            # Check if it is already wrapped. If not, wrap it.
            if full_function_path in self.original_functions:
                raise ValueError(f"{full_function_path} is already wrapped. ")

            module_path = full_function_path
            attr_chain: list[str] = []
            while True:
                try:
                    module = importlib.import_module(module_path)
                    break
                except ImportError:
                    if "." not in module_path:
                        raise
                    module_path, attr = module_path.rsplit(".", 1)
                    attr_chain.insert(0, attr)
            obj: Any = module
            for attr in attr_chain[:-1]:
                obj = getattr(obj, attr)
            parent: Any = obj
            original_function: Callable = getattr(parent, attr_chain[-1])

            self.original_functions[full_function_path] = (
                parent,
                attr_chain[-1],
                original_function,
            )
            wrapped_function: Callable = self.wrapper(
                original_function, full_function_path
            )
            setattr(parent, attr_chain[-1], wrapped_function)

        try:
            yield
        finally:
            # Restore the original function.
            for full_function_path in list(self.original_functions):
                assert full_function_path in self.original_functions
                parent, last_attr, original_function = self.original_functions.pop(
                    full_function_path
                )
                setattr(parent, last_attr, original_function)

    def get_elapsed_times(self) -> dict[str, list[float]]:
        """
        Get ALL elasped times. As elasped time is measured with torch.cuda.Event,
        torch.cuda.synchronize() must be called to get the elapsed time.
        """
        torch.cuda.synchronize()
        elasped_times = defaultdict(list)

        for event in self.events:
            full_function_path, start_event, end_event = event
            elapsed_time = start_event.elapsed_time(end_event)
            elasped_times[full_function_path].append(elapsed_time)

        return elasped_times
