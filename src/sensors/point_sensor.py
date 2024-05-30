import numpy as np

from . import BaseSensor


class PointSensor(BaseSensor):

    def sense(self, states: np.ndarray) -> np.ndarray:
        if states.ndim == 1:
            states = states.reshape(1, -1)
        noise_free = self.get(states[:, 0], states[:, 1])
        observations = self.rng.normal(loc=noise_free, scale=self.noise_scale)
        return observations.reshape(-1, 1)
