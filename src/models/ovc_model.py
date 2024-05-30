import gpytorch
import numpy as np
import torch
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
)

from ..scalers import MinMaxScaler, StandardScaler
from .poam_model import POAMModel


class GPyTorchModel(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        inducing_x,
        learn_inducing,
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
            learn_inducing_locations=learn_inducing,
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

    def update_variational(self, x, y):
        z = self.variational_strategy.inducing_points
        noise_var = self.likelihood.noise_covar.noise
        Kuf = self.covar_module(z, x).evaluate()
        Kuu = self.covar_module(z).evaluate()
        a = Kuf @ y.unsqueeze(-1) / noise_var  # a = Kuf @ invSigma @ y
        B = Kuf.matmul(Kuf.t()) / noise_var  # B = Kuf @ invSigma @ Kfu

        if hasattr(self, "old_a"):
            Kofu = self.covar_module(self.old_x, z).evaluate()
            P = self.pseudo_inverse @ Kofu
            a = a + P.t() @ self.old_a
            B = B + P.t() @ self.old_B @ P

        L = psd_safe_cholesky(Kuu + B)
        m = Kuu @ torch.cholesky_solve(a, L)
        S = Kuu @ torch.cholesky_solve(Kuu, L)
        Ls = psd_safe_cholesky(S, jitter=self.jitter)

        q_dist = self.variational_strategy._variational_distribution
        q_mean = q_dist.variational_mean
        q_mean.data = m.squeeze(-1)
        q_chol = q_dist.chol_variational_covar
        q_chol.data = Ls
        self.variational_strategy.variational_params_initialized.fill_(1)

        self.old_x = z.detach().clone()
        self.old_a = a.detach().clone()
        self.old_B = B.detach().clone()
        Kouof = self.covar_module(z, self.old_x).evaluate().detach()
        pseudo_inverse = torch.cholesky_solve(
            Kouof, psd_safe_cholesky(Kouof.matmul(Kouof.t()), jitter=self.jitter)
        )
        self.pseudo_inverse = pseudo_inverse


class OVCModel(POAMModel):
    def __init__(
        self,
        num_inducing: int,
        learn_inducing: bool,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_scaler: MinMaxScaler,
        y_scaler: StandardScaler,
        kernel: gpytorch.kernels.Kernel,
        noise_variance: float,
        batch_size: int = 128,
        jitter: float = 1e-6,
    ):
        super().__init__(
            num_inducing,
            learn_inducing,
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
        self.model = GPyTorchModel(
            self._init_inducing(),
            self.learn_inducing,
            self.kernel,
            self.likelihood,
            self.jitter,
        ).double()
        self.model.update_variational(self.train_x, self.train_y)
