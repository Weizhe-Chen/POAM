import numpy as np


class Evaluator:
    def __init__(self, sensor, task_extent, eval_grid):
        x_min, x_max, y_min, y_max = task_extent
        num_x, num_y = eval_grid
        x_grid = np.linspace(x_min, x_max, num_x)
        y_grid = np.linspace(y_min, y_max, num_y)
        xx, yy = np.meshgrid(x_grid, y_grid)
        self.eval_grid = eval_grid
        self.eval_inputs = np.column_stack((xx.flatten(), yy.flatten()))
        self.eval_outputs = sensor.get(
            self.eval_inputs[:, 0], self.eval_inputs[:, 1]
        ).reshape(-1, 1)
        self.eval_outputs_var = np.var(self.eval_outputs)
        self.num_samples = []
        self.smses = []
        self.rmses = []
        self.maes = []
        self.mslls = []
        self.nlpds = []
        self.training_times = []
        self.prediction_times = []
        self.losses = []
        self.x_train = None
        self.y_train = None

    def log_gaussian_density(self, error: np.ndarray, std: np.ndarray) -> float:
        return -0.5 * np.log(2 * np.pi) - np.log(std) - 0.5 * (error / std) ** 2

    def negative_log_predictive_density(self) -> float:
        return np.mean(-self.log_gaussian_density(self.abs_error, self.std))

    def mean_square_error(self) -> float:
        return np.mean(self.abs_error**2)

    def root_mean_square_error(self) -> float:
        return np.sqrt(self.mean_square_error())

    def standardized_mean_square_error(self) -> float:
        mse = self.mean_square_error()
        return mse / self.eval_outputs_var

    def mean_absolute_error(self) -> float:
        return np.mean(np.fabs(self.abs_error))

    def mean_standardized_log_loss(self) -> float:
        log_loss = -self.log_gaussian_density(self.abs_error, self.std)
        baseline_mean = np.mean(self.y_train)
        baseline_std = np.std(self.y_train)
        baseline_error = self.eval_outputs - baseline_mean
        baseline_log_loss = -self.log_gaussian_density(baseline_error, baseline_std)
        return np.mean(log_loss - baseline_log_loss)

    def compute_metrics(self, mean, std):
        self.mean = mean
        self.std = std
        self.abs_error = np.fabs(mean - self.eval_outputs)

        self.num_samples.append(self.y_train.shape[0])
        self.smses.append(self.standardized_mean_square_error())
        self.rmses.append(self.root_mean_square_error())
        self.maes.append(self.mean_absolute_error())
        self.mslls.append(self.mean_standardized_log_loss())
        self.nlpds.append(self.negative_log_predictive_density())

        self.mean = self.mean.reshape(self.eval_grid)
        self.std = self.std.reshape(self.eval_grid)
        self.abs_error = self.abs_error.reshape(self.eval_grid)

    def add_data(self, x_new, y_new):
        if self.x_train is None and self.y_train is None:
            self.x_train = x_new
            self.y_train = y_new
        else:
            self.x_train = np.vstack([self.x_train, x_new])
            self.y_train = np.vstack([self.y_train, y_new])
