import numpy as np
import torch
import gpytorch
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
    VariationalStrategy,
)
from .base_model import BaseModel
from .gpytorch_settings import gpytorch_settings
from ..scalers import MinMaxScaler, StandardScaler


class GPyTorchModel(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        inducing_x,
        learn_inducing,
        learn_variational,
        kernel,
        likelihood,
        jitter=1e-6,
    ):
        self.num_inducing = inducing_x.size(0)
        variational_distribution = CholeskyVariationalDistribution(self.num_inducing)
        # variational_strategy = UnwhitenedVariationalStrategy(
        if learn_variational:
            strategy = VariationalStrategy
        else:
            strategy = UnwhitenedVariationalStrategy
        variational_strategy = strategy(
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
        q_dist.variational_mean.requires_grad_(True)
        q_dist.chol_variational_covar.requires_grad_(True)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # def update_variational(self, x: torch.Tensor, y: torch.Tensor) -> None:
    #     z = self.variational_strategy.inducing_points
    #     noise_var = self.likelihood.noise_covar.noise
    #     Kuf = self.covar_module(z, x).evaluate()
    #     Kuu = self.covar_module(z).evaluate()
    #     # Data-dependent terms
    #     a = Kuf @ y.unsqueeze(-1) / noise_var
    #     B = Kuf @ Kuf.t() / noise_var
    #     # Compute variational parameters
    #     L = psd_safe_cholesky(Kuu + B, jitter=self.jitter)
    #     m = Kuu @ torch.cholesky_solve(a, L)
    #     S = Kuu @ torch.cholesky_solve(Kuu, L)
    #     # Update variational parameters
    #     Ls = psd_safe_cholesky(S, jitter=self.jitter)
    #     q_dist = self.variational_strategy._variational_distribution
    #     q_dist.variational_mean.data = m.squeeze(-1)
    #     q_dist.chol_variational_covar.data = Ls
    #     self.variational_strategy.variational_params_initialized.fill_(1)


class SVGPModel(BaseModel):
    def __init__(
        self,
        num_inducing: int,
        learn_inducing: bool,
        learn_variational: bool,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_scaler: MinMaxScaler,
        y_scaler: StandardScaler,
        kernel: gpytorch.kernels.Kernel,
        noise_variance: float,
        batch_size: int,
        jitter: float = 1e-6,
    ):
        self.num_inducing = num_inducing
        self.learn_inducing = learn_inducing
        self.learn_variational = learn_variational
        super().__init__(
            x_train,
            y_train,
            x_scaler,
            y_scaler,
            kernel,
            noise_variance,
            batch_size,
            jitter,
        )

    def _init_inducing(self) -> torch.Tensor:
        print("Init inducing inputs with random subset of training data...")
        indices = torch.randint(high=len(self.train_y), size=(self.num_inducing,))
        return self.train_x[indices].clone()

    def _init_model(self) -> None:
        print("Initializing model...")
        self.model = GPyTorchModel(
            self._init_inducing(),
            self.learn_inducing,
            self.learn_variational,
            self.kernel,
            self.likelihood,
            self.jitter,
        ).double()

    def _init_evidence(self) -> None:
        self.evidence = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=self.train_y.size(0)
        )

    def _init_optimizer(self, slow_lr: float = 1e-3, fast_lr: float = 1e-2) -> None:
        slow_params, fast_params = [], []
        print("Model parameters:")
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # don't optimize variational params
            if not self.learn_variational and "variational_distribution" in name:
                continue
            if "nn" in name:
                slow_params.append(param)
                print(f"Slow lr: {slow_lr}", name, param.shape)
            else:
                fast_params.append(param)
                print(f"Fast lr: {fast_lr}", name, param.shape)
        self.optimizer = torch.optim.Adam(
            [
                {"params": slow_params, "lr": slow_lr},
                {"params": fast_params, "lr": fast_lr},
            ],
            lr=slow_lr,
        )

    def add_data(self, x: np.ndarray, y: np.ndarray) -> None:
        self.new_x = self._preprocess_x(x)
        self.new_y = self._preprocess_y(y)
        self.train_x = torch.cat([self.train_x, self.new_x])
        self.train_y = torch.cat([self.train_y, self.new_y])
        self.evidence.num_data = self.train_y.size(0)

    def optimize(self, num_steps: int):
        self.model.train()
        self.likelihood.train()
        losses = []
        with gpytorch_settings():
            for i in range(num_steps):
                batch = torch.randint(high=len(self.train_y), size=(self.batch_size,))
                batch_x = self.train_x[batch]
                batch_y = self.train_y[batch]
                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = -self.evidence(output, batch_y)
                print(f"Iter {i:04d} | ELBO: {loss.item(): .2f} ")
                loss.backward()
                losses.append(loss.item())
                self.optimizer.step()
        self.model.eval()
        self.likelihood.eval()
        return losses

    @property
    @torch.no_grad()
    def x_inducing(self):
        x_inducing = self.model.variational_strategy.inducing_points.numpy()
        x_inducing = self.x_scaler.postprocess(x_inducing)
        return x_inducing

    @torch.no_grad()
    def get_ak_lengthscales(self, x):
        list_features = []
        x = self._preprocess_x(x)
        for x_batch in torch.split(x, self.batch_size):
            features = self.kernel.base_kernel.get_features(x_batch)
            list_features.append(features.numpy())
        features = np.vstack(list_features)
        features = features / features.sum(axis=1, keepdims=True)
        primitive_lengthscales = self.kernel.base_kernel.lengthscales.numpy()
        lengthscales = features @ primitive_lengthscales.reshape(-1, 1)
        return lengthscales
