from abc import ABCMeta, abstractmethod
from typing import List
import numpy as np


class BasePlanner(metaclass=ABCMeta):
    def __init__(self, task_extent: List[float], rng: np.random.RandomState) -> None:
        self.task_extent = task_extent
        self.rng = rng

    @abstractmethod
    def plan(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError
