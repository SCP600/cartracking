from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VideoSource(ABC):
    frame_index: int = 0
    timestamp_ms: float = 0.0
    frame_count: int = 0
    fps: float = 0.0

    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        raise NotImplementedError

    @abstractmethod
    def release(self) -> None:
        raise NotImplementedError

    def seek(self, frame_index: int) -> bool:
        return False
