import gpytorch
import numpy as np
import torch
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
)

from ..scalers import MinMaxScaler, StandardScaler
from .base_model import BaseModel
from .gpytorch_settings import gpytorch_settings


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
        Kuf_y = Kuf @ y.unsqueeze(-1) / noise_var
        Kuf_Kfu = Kuf.matmul(Kuf.t()) / noise_var

        if hasattr(self, "old_z"):
            Kouu = self.covar_module(self.old_z, z).evaluate()
            invS_Kouu = torch.cholesky_solve(Kouu, self.old_Ls)
            Kuou_invS_m = invS_Kouu.t() @ self.old_m
            Kuf_y = Kuf_y + Kuou_invS_m

            Kuou_invS_Kouu = invS_Kouu.t() @ Kouu
            Lk = psd_safe_cholesky(self.old_K, jitter=self.jitter)
            invK_Kouu = torch.cholesky_solve(Kouu, Lk)
            Kuou_invK_Kouu = Kouu.t() @ invK_Kouu
            Kuf_Kfu = Kuf_Kfu + Kuou_invS_Kouu - Kuou_invK_Kouu

        L = psd_safe_cholesky(Kuu + Kuf_Kfu)
        m = Kuu @ torch.cholesky_solve(Kuf_y, L)
        S = Kuu @ torch.cholesky_solve(Kuu, L)
        Ls = psd_safe_cholesky(S, jitter=self.jitter)

        q_dist = self.variational_strategy._variational_distribution
        q_mean = q_dist.variational_mean
        q_mean.data = m.squeeze(-1)
        q_chol = q_dist.chol_variational_covar
        q_chol.data = Ls
        self.variational_strategy.variational_params_initialized.fill_(1)

        self.old_z = z.detach().clone()
        self.old_m = m.detach().clone()
        self.old_Ls = Ls.detach().clone()
        self.old_K = Kuu.detach().clone()

    def online_elbo(self, x, y):
        num_train = x.size(0)
        z = self.variational_strategy.inducing_points
        noise_var = self.likelihood.noise_covar.noise

        Kvv = self.covar_module(z).add_jitter(self.jitter).evaluate()
        Kvf = self.covar_module(z, x).evaluate()
        Kvu = self.covar_module(z, self.old_z).evaluate()
        Kuu = self.covar_module(self.old_z).evaluate()
        kff = self.covar_module(x, diag=True)

        TrilKvv = psd_safe_cholesky(Kvv, jitter=self.jitter)
        TrilSuu = self.old_Ls
        TrilOldKuu = psd_safe_cholesky(self.old_K, jitter=self.jitter)

        InvTrilKvv_Kvf = torch.linalg.solve_triangular(TrilKvv, Kvf, upper=False)
        D1 = (InvTrilKvv_Kvf @ InvTrilKvv_Kvf.T).div(noise_var)
        InvTrilKvv_Kvu = torch.linalg.solve_triangular(TrilKvv, Kvu, upper=False)
        Kuv_InvTrilKvvT = InvTrilKvv_Kvu.T
        InvTrilSuu_Kuv_InvTrilKvvT = torch.linalg.solve_triangular(
            TrilSuu, Kuv_InvTrilKvvT, upper=False
        )
        D2 = InvTrilSuu_Kuv_InvTrilKvvT.T @ InvTrilSuu_Kuv_InvTrilKvvT
        InvTrilOldKuu_Kuv_InvTrilKvvT = torch.linalg.solve_triangular(
            TrilOldKuu, Kuv_InvTrilKvvT, upper=False
        )
        D3 = InvTrilOldKuu_Kuv_InvTrilKvvT.T @ InvTrilOldKuu_Kuv_InvTrilKvvT
        D = torch.eye(D1.shape[0]).to(D1) + D1 + D2 - D3
        TrilD = psd_safe_cholesky(D, jitter=self.jitter)

        InvTrilSuu_Kuv = torch.linalg.solve_triangular(TrilSuu, Kvu.T, upper=False)
        InvTrilSuu_mu = torch.linalg.solve_triangular(TrilSuu, self.old_m, upper=False)
        c1 = (Kvf @ y.view(-1, 1)).div(noise_var)
        c2 = InvTrilSuu_Kuv.T @ InvTrilSuu_mu
        c = c1 + c2

        InvTrilKvv_c = torch.linalg.solve_triangular(TrilKvv, c, upper=False)
        InvTrilD_InvTrilKvv_c = torch.linalg.solve_triangular(
            TrilD, InvTrilKvv_c, upper=False
        )
        InvTrilSuu_mu = torch.linalg.solve_triangular(TrilSuu, self.old_m, upper=False)

        InvTrilKvv_Kvu = torch.linalg.solve_triangular(TrilKvv, Kvu, upper=False)
        Quu = InvTrilKvv_Kvu.T @ InvTrilKvv_Kvu
        Euu = Kuu - Quu
        InvSuu_Euu = torch.cholesky_solve(Euu, TrilSuu)
        InvOldKuu_Euu = torch.cholesky_solve(Euu, TrilOldKuu)
        InvTrilKvv_Kvf = torch.linalg.solve_triangular(TrilKvv, Kvf, upper=False)

        constant_term = -num_train * np.log(2 * np.pi)
        quadratic_terms = 0.0
        quadratic_terms = (
            -(y.square().sum().div(noise_var))
            + InvTrilD_InvTrilKvv_c.square().sum()
            - InvTrilSuu_mu.square().sum()  # constant
        )
        log_terms = (
            -TrilD.diagonal().square().log().sum()
            - TrilSuu.diagonal().square().log().sum()  # constant
            + TrilOldKuu.diagonal().square().log().sum()  # constant
            - num_train * torch.log(noise_var)  # constant
        )
        trace_terms = (
            -InvSuu_Euu.trace()
            + InvOldKuu_Euu.trace()
            - kff.sum().div(noise_var)
            + InvTrilKvv_Kvf.square().sum().div(noise_var)
        )
        elbo = 0.5 * (constant_term + quadratic_terms + log_terms + trace_terms)
        return elbo


class SSGPModel(BaseModel):
    def __init__(
        self,
        num_inducing: int,
        learn_inducing: bool,
        strategy_inducing: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_scaler: MinMaxScaler,
        y_scaler: StandardScaler,
        kernel: gpytorch.kernels.Kernel,
        noise_variance: float,
        batch_size: int = 128,
        jitter: float = 1e-6,
    ):
        self.num_inducing = num_inducing
        self.learn_inducing = learn_inducing
        self.strategy_inducing = strategy_inducing
        self.first_update = True
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

    def _init_optimizer(self, slow_lr: float = 1e-3, fast_lr: float = 1e-2) -> None:
        slow_params, fast_params = [], []
        print("Model parameters:")
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # don't optimize variational params
            if "variational_distribution" in name:
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

    def _init_inducing(self) -> torch.Tensor:
        indices = torch.randint(high=len(self.train_y), size=(self.num_inducing,))
        return self.train_x[indices].clone()

    def _init_model(self) -> None:
        self.model = GPyTorchModel(
            self._init_inducing(),
            self.learn_inducing,
            self.kernel,
            self.likelihood,
            self.jitter,
        ).double()
        self.model.update_variational(self.train_x, self.train_y)

    def _init_evidence(self) -> None:
        self.evidence = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=self.train_y.size(0)
        )

    def add_data(self, x_new: np.ndarray, y_new: np.ndarray):
        self.train_x = self._preprocess_x(x_new)
        self.train_y = self._preprocess_y(y_new)
        if self.first_update:
            self.first_update = False

    @torch.no_grad()
    def update_inducing(self) -> torch.Tensor:
        print("Updating inducing inputs...")
        if self.strategy_inducing == "cholesky":
            old_z = self.model.variational_strategy.inducing_points.data
            candidates = torch.cat([old_z, self.train_x], dim=0)
            indices = gpytorch.pivoted_cholesky(
                self.kernel(candidates), rank=self.num_inducing, return_pivots=True
            )[1][: self.num_inducing]
            self.model.variational_strategy.inducing_points.data = candidates[indices]
        elif self.strategy_inducing == "random":
            num_old = int(self.num_inducing * 0.99)
            num_new = self.num_inducing - num_old
            old_z = self.model.variational_strategy.inducing_points.data
            old_indices = torch.randint(high=len(old_z), size=(num_old,))
            new_indices = torch.randint(high=len(self.train_x), size=(num_new,))
            z = torch.cat([old_z[old_indices], self.train_x[new_indices]], dim=0)
            self.model.variational_strategy.inducing_points.data = z
        else:
            raise ValueError("Invalid strategy for inducing inputs")

    def update_variational(self) -> None:
        print("Updating variational parameters...")
        self.model.update_variational(self.train_x, self.train_y)

    def optimize(self, num_steps: int):
        print("Updating hyper-parameters...")
        self.model.train()
        self.likelihood.train()
        losses = []
        with gpytorch_settings():
            for i in range(num_steps):
                self.optimizer.zero_grad()
                if self.first_update:
                    output = self.model(self.train_x)
                    loss = -self.evidence(output, self.train_y)
                    print(f"Iter {i:04d} | ELBO: {loss.item(): .2f} ")
                else:
                    loss = -self.model.online_elbo(self.train_x, self.train_y)
                    print(f"Iter {i:04d} | Online ELBO: {loss.item(): .2f} ")
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
