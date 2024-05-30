import numpy as np
from typing import List
from . import BasePlanner


class LawnmowerPlanner(BasePlanner):
    def __init__(self, task_extent: List[float], rng: np.random.RandomState) -> None:
        super().__init__(task_extent, rng)

    def plan(self, num_points: int, *args, **kwargs) -> np.ndarray:
        xmin, xmax, ymin, ymax = self.task_extent
        xs = np.linspace(xmin, xmax, num_points // 2)
        waypoints = []
        for x in xs:
            waypoints.append([x, ymin])
            waypoints.append([x, ymax])
        waypoints = np.array(waypoints)
        return waypoints
