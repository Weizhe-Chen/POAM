from bezier import Curve
import numpy as np
from typing import List
from . import BasePlanner


class BezierPlanner(BasePlanner):
    def __init__(self, task_extent: List[float], rng: np.random.RandomState) -> None:
        super().__init__(task_extent, rng)
        self._init_control_points()
        self._init_bezier_curve(self.control_points)

    def plan(self, num_points: int, *args, **kwargs) -> np.ndarray:
        curve_params = np.linspace(0, 1, num_points)
        waypoints = self.curve.evaluate_multi(curve_params).T
        return waypoints

    def _init_control_points(self):
        xmin, xmax, ymin, ymax = self.task_extent
        xhalf = (xmax - xmin) / 2 + xmin
        xquater = (xmax - xmin) / 4 + xmin
        xquater_x3 = 3 * (xmax - xmin) / 4 + xmin
        yhalf = (ymax - ymin) / 2 + ymin
        yquater = (ymax - ymin) / 4 + ymin
        yquater_x3 = 3 * (ymax - ymin) / 4 + ymin

        self.control_points = np.array(
            [
                [xmax, ymin],
                [xmin, ymin],
                [xmin, yquater],
                [xmin, yhalf],
                [xmin, yquater_x3],
                [xmin, ymax],
                [xquater, ymax],
                [xhalf, ymax],
                [xquater_x3, ymax],
                [xmax, ymax],
                [xmax, yquater_x3],
                [xmax, yhalf],
                [xmax, yquater],
                [xhalf, yquater],
                [xhalf, yhalf],
            ]
        ).T

    def _init_bezier_curve(self, control_points: np.ndarray):
        nodes = np.asfortranarray(control_points)
        self.curve = Curve(nodes, degree=nodes.shape[1] - 1)

