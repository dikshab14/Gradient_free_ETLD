
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable
from .mmd import mmd2_and_g_unbiased
from .kernels import GaussianKernel

@dataclass
class GFETLDConfig:
    step_size: float = 1e-3
    n_steps: int = 200
    ensemble_size: int = 100
    n_sim_per_particle: int = 20
    kernel_bandwidth: float = 1.0
    temperature: float = 1.0

class GFETLD:
    def __init__(self,
                 dim_param: int,
                 prior_loggrad: Callable[[np.ndarray], np.ndarray],
                 simulator: Callable[[np.ndarray, int], np.ndarray],
                 obs_samples: np.ndarray,
                 config: GFETLDConfig):
        self.D = dim_param
        self.prior_loggrad = prior_loggrad
        self.simulator = simulator
        self.obs = np.atleast_2d(obs_samples)
        self.cfg = config
        self.kernel = GaussianKernel(config.kernel_bandwidth)

    def _sqrtm_cholesky(self, C: np.ndarray) -> np.ndarray:
        eps = 1e-9
        return np.linalg.cholesky(C + eps * np.eye(C.shape[0]))

    def _simulate_particle_outputs(self, theta: np.ndarray) -> np.ndarray:
        J = self.cfg.n_sim_per_particle
        return np.atleast_2d(self.simulator(theta, J))

    def run(self, theta0: np.ndarray) -> dict:
        M, D = theta0.shape
        assert D == self.D
        theta = theta0.copy()
        history = {"theta_mean": [], "mmd2_mean": []}

        for t in range(self.cfg.n_steps):
            outputs = [self._simulate_particle_outputs(theta[m]) for m in range(M)]
            J, d = outputs[0].shape
            X = np.stack([out.reshape(-1) for out in outputs], axis=0)
            Theta = theta

            theta_mean = Theta.mean(axis=0)
            x_mean = X.mean(axis=0)

            dtheta = Theta - theta_mean
            dx = X - x_mean

            C_theta = (dtheta.T @ dtheta) / M
            C_theta_x = (dtheta.T @ dx) / M

            L = self._sqrtm_cholesky(C_theta)

            grad_logprior = self.prior_loggrad(theta)

            mmd2_vals = np.zeros(M, dtype=float)
            g_mat = np.zeros((M, J * d), dtype=float)
            for m in range(M):
                mmd2, g = mmd2_and_g_unbiased(outputs[m], self.obs, self.kernel)
                mmd2_vals[m] = mmd2
                g_mat[m] = g

            beta = self.cfg.temperature
            ds = self.cfg.step_size

            drifts = (grad_logprior @ C_theta.T) - (g_mat @ C_theta_x.T) * beta
            correction = ((self.D + 1.0) / M) * (theta - theta_mean)
            noise = np.random.randn(M, self.D) @ L.T * np.sqrt(2.0 * ds)

            theta = theta + (drifts + correction) * ds + noise

            history["theta_mean"].append(theta.mean(axis=0))
            history["mmd2_mean"].append(mmd2_vals.mean())

        history["theta_final"] = theta
        history["theta_mean"] = np.array(history["theta_mean"])
        history["mmd2_mean"] = np.array(history["mmd2_mean"])
        return history
