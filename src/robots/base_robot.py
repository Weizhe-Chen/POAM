from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np

from ..sensors import BaseSensor


class BaseRobot(metaclass=ABCMeta):
    def __init__(
        self,
        sensor: BaseSensor,
        state: np.ndarray,
        control_rate: float,
        max_lin_vel: float,
        max_ang_vel: float,
        goal_radius: float,
    ) -> None:
        self.sensor = sensor
        self.state = state
        self.control_dt = 1.0 / control_rate
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        self.goal_radius = goal_radius
        self.sampled_locations = []
        self.sampled_observations = []
        self.goals = []
        self.timer = 0.0

    @abstractmethod
    def compute_action(self) -> Tuple[np.ndarray, float]:
        raise NotImplementedError

    @abstractmethod
    def update_state(self, action) -> None:
        raise NotImplementedError

    def step(self) -> None:
        action, dist = self.compute_action()
        if dist < self.goal_radius:
            self.goals = self.goals[1:]
        self.update_state(action)
        self.timer += self.control_dt
        if self.timer > self.sensor.dt:
            self.sampled_locations.append([self.state[0], self.state[1]])
            self.sampled_observations.append(self.sensor.sense(self.state))
            self.timer = 0.0


    def commit_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the sampling locations."""
        locations = np.vstack(self.sampled_locations)
        observations = np.vstack(self.sampled_observations)
        self.sampled_locations = []
        self.sampled_observations = []
        return locations, observations

    @property
    def has_goals(self) -> bool:
        return len(self.goals) > 0

