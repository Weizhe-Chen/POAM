import numpy as np


class StandardScaler:
    def __init__(self, actual_mean=None, actual_std=None) -> None:
        self.actual_mean = actual_mean
        self.actual_std = actual_std

    def fit(self, x: np.ndarray) -> None:
        self.actual_mean = x.mean(axis=0, keepdims=True)
        self.actual_std = x.std(axis=0, keepdims=True)
        if np.any(self.actual_std == 0):
            raise ValueError("Standard deviation is zero for at least one dimension.")

    def preprocess(self, raw: np.ndarray) -> np.ndarray:
        assert self.actual_std is not None
        return (raw - self.actual_mean) / self.actual_std

    def postprocess_mean(self, scaled: np.ndarray) -> np.ndarray:
        return scaled * self.actual_std + self.actual_mean

    def postprocess_std(self, scaled: np.ndarray) -> np.ndarray:
        return scaled * self.actual_std
