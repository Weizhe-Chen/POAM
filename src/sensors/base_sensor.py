from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np


class BaseSensor(metaclass=ABCMeta):
    def __init__(
        self,
        matrix: np.ndarray,
        env_extent: List[float],
        rate: float,
        noise_scale: float,
        rng: np.random.RandomState,
    ) -> None:
        self.matrix = matrix
        self.env_extent = env_extent
        self.num_rows, self.num_cols = matrix.shape
        self.x_cell_size = (env_extent[1] - env_extent[0]) / self.num_cols
        self.y_cell_size = (env_extent[3] - env_extent[2]) / self.num_rows
        self.dt = 1.0 / rate
        self.noise_scale = noise_scale
        self.rng = rng

    @abstractmethod
    def sense(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def xs_to_cols(self, xs: np.ndarray) -> np.ndarray:
        cols = ((xs - self.env_extent[0]) / self.x_cell_size).astype(int)
        np.clip(cols, 0, self.num_cols - 1, out=cols)
        return cols

    def ys_to_rows(self, ys: np.ndarray) -> np.ndarray:
        rows = ((ys - self.env_extent[2]) / self.y_cell_size).astype(int)
        np.clip(rows, 0, self.num_rows - 1, out=rows)
        return rows

    def get(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        cols = self.xs_to_cols(xs)
        rows = self.ys_to_rows(ys)
        values = self.matrix[rows, cols]
        return values

    def set(self, xs: np.ndarray, ys: np.ndarray, values: np.ndarray) -> None:
        cols = self.xs_to_cols(xs)
        rows = self.ys_to_rows(ys)
        self.matrix[rows, cols] = values
