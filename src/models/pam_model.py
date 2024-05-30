import numpy as np
import torch
import gpytorch
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
)
from . import SVGPModel
from ..scalers import MinMaxScaler, StandardScaler


class GPyTorchModel(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        inducing_x,
        kernel,
        likelihood,
        jitter=1e-6,
    ):
        self.num_inducing = inducing_x.size(0)
        variational_distribution = CholeskyVariationalDistribution(self.num_inducing)
        variational_strategy = UnwhitenedVariationalStrategy(
            self,
            inducing_x,
            variational_distribution,
            learn_inducing_locations=False,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel
        self.likelihood = likelihood
        self.jitter = jitter
        q_dist = self.variational_strategy._variational_distribution
        q_dist.variational_mean.requires_grad_(False)
        q_dist.chol_variational_covar.requires_grad_(False)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def update_variational(self, x: torch.Tensor, y: torch.Tensor) -> None:
        z = self.variational_strategy.inducing_points
        noise_var = self.likelihood.noise_covar.noise
        Kuf = self.covar_module(z, x).evaluate()
        Kuu = self.covar_module(z).evaluate()
        # Data-dependent terms
        a = Kuf @ y.unsqueeze(-1) / noise_var
        B = Kuf @ Kuf.t() / noise_var
        # Compute variational parameters
        L = psd_safe_cholesky(Kuu + B, jitter=self.jitter)
        m = Kuu @ torch.cholesky_solve(a, L)
        S = Kuu @ torch.cholesky_solve(Kuu, L)
        # Update variational parameters
        Ls = psd_safe_cholesky(S, jitter=self.jitter)
        q_dist = self.variational_strategy._variational_distribution
        q_dist.variational_mean.data = m.squeeze(-1)
        q_dist.chol_variational_covar.data = Ls
        self.variational_strategy.variational_params_initialized.fill_(1)


class PAMModel(SVGPModel):
    def __init__(
        self,
        num_inducing: int,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_scaler: MinMaxScaler,
        y_scaler: StandardScaler,
        kernel: gpytorch.kernels.Kernel,
        noise_variance: float,
        batch_size: int,
        jitter: float = 1e-6,
    ):
        super().__init__(
            num_inducing,
            False,
            False,
            x_train,
            y_train,
            x_scaler,
            y_scaler,
            kernel,
            noise_variance,
            batch_size,
            jitter,
        )

    def _init_model(self) -> None:
        print("Initializing model...")
        self.model = GPyTorchModel(
            self._init_inducing(),
            self.kernel,
            self.likelihood,
            self.jitter,
        ).double()

    def update_inducing(self) -> torch.Tensor:
        print("Updating inducing inputs...")
        indices = gpytorch.pivoted_cholesky(
            self.kernel(self.train_x), rank=self.num_inducing, return_pivots=True
        )[1][: self.num_inducing]
        new_inducing = self.train_x[indices].clone()
        self.model.variational_strategy.inducing_points.data = new_inducing

    def update_variational(self) -> None:
        print("Updating variational parameters...")
        self.model.update_variational(self.train_x, self.train_y)
