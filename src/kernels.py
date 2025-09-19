
import numpy as np

class GaussianKernel:
    def __init__(self, bandwidth: float):
        self.sigma2 = float(bandwidth) ** 2

    def pairwise(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        diff = X[:, None, :] - Y[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        return np.exp(-0.5 * dist2 / self.sigma2)

    def grad_wrt_x(self, x: np.ndarray, Y: np.ndarray) -> np.ndarray:
        x = x.reshape(1, -1)
        Y = np.atleast_2d(Y)
        Kxy = self.pairwise(x, Y).reshape(-1, 1)
        return (Kxy * (Y - x)) / self.sigma2
