from contextlib import contextmanager

from gpytorch.settings import (
    debug,
    fast_computations,
    fast_pred_var,
    lazily_evaluate_kernels,
    max_cholesky_size,
    max_eager_kernel_size,
    memory_efficient,
    sgpr_diagonal_correction,
    skip_logdet_forward,
    skip_posterior_variances,
    verbose_linalg,
)


@contextmanager
def gpytorch_settings():
    # lazily_evaluate_kernels(True) is important for numerical stability of SGPR + AK
    with debug(False) as d, fast_computations(False, False, False) as fc, fast_pred_var(
        False
    ) as fpv, lazily_evaluate_kernels(True) as lek, max_cholesky_size(
        1e8
    ) as mcs, max_eager_kernel_size(
        1e8
    ) as meks, memory_efficient(
        False
    ) as me, sgpr_diagonal_correction(
        False
    ) as sdc, skip_logdet_forward(
        False
    ) as slf, skip_posterior_variances(
        False
    ) as spv, verbose_linalg(
        False
    ) as vl:
        yield (d, fc, fpv, lek, mcs, meks, me, sdc, slf, spv, vl)
