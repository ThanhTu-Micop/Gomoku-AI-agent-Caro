from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):
    @abstractmethod
    def get_move(self, grid: np.ndarray, player: int) -> tuple[int, int]:
        ...
