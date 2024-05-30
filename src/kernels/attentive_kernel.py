from typing import Union

import gpytorch
import torch
import torch.nn as nn
from linear_operator.operators import LinearOperator
from torch import Tensor


class AttentiveKernel(gpytorch.kernels.Kernel):
    is_stationary = False

    def __init__(
        self,
        dim_input: int,
        dim_hidden: int = 8,
        dim_output: int = 8,
        min_lengthscale: float = 0.01,
        max_lengthscale: float = 0.5,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.nn = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_output),
            nn.Softmax(dim=1),
        )
        self.lengthscales = torch.linspace(min_lengthscale, max_lengthscale, dim_output)

    def get_features(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        z = self.nn(x)
        return z / z.norm(dim=1, keepdim=True)

    def _base_kernel(self, dist, lengthscale):
        return torch.exp(-0.5 * torch.square(dist / lengthscale))

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params
    ) -> Union[Tensor, LinearOperator]:
        if diag:
            return torch.ones(len(x1)).to(x1)
        z1 = self.get_features(x1)
        z2 = self.get_features(x2)
        dist = self.covar_dist(x1, x2, **params)
        covar = torch.zeros_like(dist)
        for i in range(len(self.lengthscales)):
            lengthscale_attention = torch.outer(z1[:, i], z2[:, i])
            covar += lengthscale_attention * self._base_kernel(
                dist, self.lengthscales[i]
            )
        similarity_attention = z1 @ z2.t()
        covar *= similarity_attention
        return covar
