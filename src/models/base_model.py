from typing import Tuple
from abc import ABCMeta, abstractmethod

import gpytorch
import numpy as np
import torch

from ..scalers import MinMaxScaler, StandardScaler
from .gpytorch_settings import gpytorch_settings


class BaseModel(metaclass=ABCMeta):
    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_scaler: MinMaxScaler,
        y_scaler: StandardScaler,
        kernel: gpytorch.kernels.Kernel,
        noise_variance: float,
        batch_size: int,
        jitter: float = 1e-6,
    ):
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.train_x = self._preprocess_x(x_train)
        self.train_y = self._preprocess_y(y_train)
        self.kernel = kernel.double()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        self.likelihood.noise_covar.noise = noise_variance
        self.batch_size = batch_size
        self.jitter = jitter
        self._init_model()
        self._init_evidence()
        self._init_optimizer()


    def _preprocess_x(self, x: np.ndarray) -> torch.Tensor:
        x = torch.tensor(self.x_scaler.preprocess(x), dtype=torch.float64)
        return x

    def _preprocess_y(self, y: np.ndarray) -> torch.Tensor:
        y = torch.tensor(self.y_scaler.preprocess(y), dtype=torch.float64).squeeze(-1)
        return y

    @abstractmethod
    def _init_optimizer(self, slow_lr: float = 1e-3, fast_lr: float = 1e-2) -> None:
        raise NotImplementedError

    @abstractmethod
    def _init_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _init_evidence(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_data(self, x: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def predict(
        self, x: np.ndarray, without_likelihood: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        means = []
        stds = []
        self.model.train()
        self.likelihood.train()
        self.model.eval()
        self.likelihood.eval()
        x = self._preprocess_x(x)
        with gpytorch_settings():
            for x_batch in torch.split(x, self.batch_size):
                predictive_dist = self.model(x_batch)
                if not without_likelihood:
                    predictive_dist = self.model.likelihood(predictive_dist)
                mean = predictive_dist.mean.numpy().reshape(-1, 1)
                std = predictive_dist.stddev.numpy().reshape(-1, 1)
                means.append(mean)
                stds.append(std)
        mean = np.vstack(means)
        std = np.vstack(stds)
        mean = self.y_scaler.postprocess_mean(mean)
        std = self.y_scaler.postprocess_std(std)
        return mean, std
