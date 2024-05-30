import gpytorch
import numpy as np
import torch
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
    VariationalStrategy,
)
from sklearn.cluster import kmeans_plusplus
from src.models import POAMModel
from src.scalers import MinMaxScaler, StandardScaler
from src.models import gpytorch_settings


class GPyTorchModel(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        inducing_x,
        learn_inducing,
        learn_variational,
        kernel,
        likelihood,
        jitter=1e-6,
        use_online_elbo: bool =False,
    ):
        self.num_inducing = inducing_x.size(0)
        self.use_online_elbo = use_online_elbo
        variational_distribution = CholeskyVariationalDistribution(self.num_inducing)
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
        q_dist.variational_mean.requires_grad_(learn_variational)
        q_dist.chol_variational_covar.requires_grad_(learn_variational)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def _update_variational_ssgp(self, x, y):
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

    def _update_variational_poam(self, x, y):
        z = self.variational_strategy.inducing_points
        noise_var = self.likelihood.noise_covar.noise
        Kuf = self.covar_module(z, x).evaluate()
        Kuu = self.covar_module(z).evaluate()
        a = Kuf @ y.unsqueeze(-1)  # a = Kuf @ y
        B = Kuf.matmul(Kuf.t())  # B = Kuf @ Kfu

        if hasattr(self, "old_a"):
            Kofu = self.covar_module(self.old_x, z).evaluate()
            P = self.pseudo_inverse @ Kofu
            a = a + P.t() @ self.old_a
            B = B + P.t() @ self.old_B @ P

        L = psd_safe_cholesky(Kuu + B / noise_var)
        m = Kuu @ torch.cholesky_solve(a, L) / noise_var
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

        if self.use_online_elbo:
            self.old_z = z.detach().clone()
            self.old_m = m.detach().clone()
            self.old_Ls = Ls.detach().clone()
            self.old_K = Kuu.detach().clone()

    def update_variational(self, x, y, method="poam"):
        if method == "ssgp":
            self._update_variational_ssgp(x, y)
        elif method == "poam":
            self._update_variational_poam(x, y)
        else:
            raise ValueError(f"Invalid method: {method}")

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


class AblationModel(POAMModel):
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
        batch_size: int = 128,
        jitter: float = 1e-6,
        use_online_elbo: bool = False,
    ):
        self.learn_variational = learn_variational
        self.use_online_elbo = use_online_elbo
        if self.use_online_elbo:
            self.first_update = True
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

    def add_data(self, x_new: np.ndarray, y_new: np.ndarray):
        super().add_data(x_new, y_new)
        if self.use_online_elbo and self.first_update:
            self.first_update = False

    def optimize(self, num_steps: int):
        print("Updating hyper-parameters...")
        self.model.train()
        self.likelihood.train()
        losses = []
        with gpytorch_settings():
            for i in range(num_steps):
                self.optimizer.zero_grad()
                if self.use_online_elbo and not self.first_update:
                    loss = -self.model.online_elbo(self.new_x, self.new_y)
                    print(f"Iter {i:04d} | Online ELBO: {loss.item(): .2f} ")
                else:
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

    @torch.no_grad()
    def update_inducing(self, name: str) -> torch.Tensor:
        print("Updating inducing inputs...")
        if name == "poam-z-opt":
            return
        elif name == "poam-z-rand":
            old_z = self.model.variational_strategy.inducing_points.data
            candidates = torch.cat([old_z, self.new_x], dim=0)
            indices = torch.randint(high=len(candidates), size=(self.num_inducing,))
            self.model.variational_strategy.inducing_points.data = candidates[indices]
        elif name == "poam-z-kmeans":
            old_z = self.model.variational_strategy.inducing_points.data
            candidates = torch.cat([old_z, self.new_x], dim=0).cpu().numpy()
            centroids, _ = kmeans_plusplus(candidates, self.num_inducing)
            z = torch.tensor(centroids, dtype=torch.float64)
            self.model.variational_strategy.inducing_points.data = z
        else:
            old_z = self.model.variational_strategy.inducing_points.data
            candidates = torch.cat([old_z, self.new_x], dim=0)
            indices = gpytorch.pivoted_cholesky(
                self.kernel(candidates), rank=self.num_inducing, return_pivots=True
            )[1][: self.num_inducing]
            self.model.variational_strategy.inducing_points.data = candidates[indices]

    def update_variational(self, method: str="poam") -> None:
        print("Updating variational parameters...")
        self.model.update_variational(self.new_x, self.new_y, method)

    def _init_optimizer(self, slow_lr: float = 1e-3, fast_lr: float = 1e-2) -> None:
        slow_params, fast_params = [], []
        print("Model parameters:")
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
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

    def _init_model(self) -> None:
        self.model = GPyTorchModel(
            self._init_inducing(),
            self.learn_inducing,
            self.learn_variational,
            self.kernel,
            self.likelihood,
            self.jitter,
            self.use_online_elbo,
        ).double()
        self.model.update_variational(self.train_x, self.train_y)
