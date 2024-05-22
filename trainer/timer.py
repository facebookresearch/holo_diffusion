import time
import torch
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Optional


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


@dataclass
class Timer(ContextDecorator):
    """Time your code using a class, context manager, or decorator"""

    timers: ClassVar[Dict[str, float]] = {}
    timers_n_ticks: ClassVar[Dict[str, float]] = {}
    name: str = "Timer"
    text: str = "{:s}: Elapsed time: {:0.4f} seconds"
    logger: Optional[Callable[[str], None]] = print
    cuda_sync: bool = False
    disable: bool = False
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialization: add timer to dict of timers"""
        if self.name:
            self.timers.setdefault(self.name, 0)
            self.timers_n_ticks.setdefault(self.name, 0)

    def start(self) -> None:
        """Start a new timer"""
        if self.disable:
            return
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        if self.cuda_sync:
            torch.cuda.synchronize()
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self.disable:
            return None
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        if self.cuda_sync:
            torch.cuda.synchronize()
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(self.name, elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time
            self.timers_n_ticks[self.name] += 1

        return elapsed_time

    def __enter__(self) -> "Timer":
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the context manager timer"""
        self.stop()
