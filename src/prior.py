
import numpy as np

class GaussianPrior:
    def __init__(self, mu, var):
        self.mu = np.asarray(mu, dtype=float)
        self.var = np.asarray(var, dtype=float)
        assert self.mu.shape == self.var.shape

    def sample(self, size: int):
        return self.mu + np.sqrt(self.var) * np.random.randn(size, self.mu.size)

    def logpdf_grad(self, theta: np.ndarray) -> np.ndarray:
        return -(theta - self.mu) / self.var
