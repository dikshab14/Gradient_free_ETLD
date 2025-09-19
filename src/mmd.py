
import numpy as np
from .kernels import GaussianKernel

def mmd2_and_g_unbiased(X: np.ndarray, Y: np.ndarray, kernel: GaussianKernel):
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    J, d = X.shape
    N = Y.shape[0]
    use_unbiased = (J >= 2 and N >= 2)

    Kxx = kernel.pairwise(X, X)
    Kyy = kernel.pairwise(Y, Y)
    Kxy = kernel.pairwise(X, Y)

    if use_unbiased:
        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)
        mmd2 = Kxx.sum() / (J * (J - 1)) + Kyy.sum() / (N * (N - 1)) - 2.0 * Kxy.mean()
    else:
        mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()

    g_list = []
    for j in range(J):
        xj = X[j]
        if J >= 2:
            idx = np.arange(J) != j
            grads_xj_xl = kernel.grad_wrt_x(xj, X[idx])
            term1 = (2.0 / (J * (J - 1))) * grads_xj_xl.sum(axis=0)
        else:
            grads_xj_xl = kernel.grad_wrt_x(xj, X)
            term1 = (2.0 / (max(J,1)**2)) * grads_xj_xl.sum(axis=0)

        grads_xj_yn = kernel.grad_wrt_x(xj, Y)
        term2 = (2.0 / (max(J,1) * max(N,1))) * grads_xj_yn.sum(axis=0)

        g_list.append(term1 - term2)

    g = np.concatenate(g_list, axis=0)
    return float(mmd2), g
