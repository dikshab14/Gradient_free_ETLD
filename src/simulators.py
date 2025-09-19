
import numpy as np

def gaussian_location_generator(theta: float, size: int) -> np.ndarray:
    return (float(theta) + np.random.randn(size)).reshape(-1, 1)

def uniform_location_generator(theta: float, size: int) -> np.ndarray:
    return (float(theta) - 1.0 + 2.0 * np.random.rand(size)).reshape(-1, 1)

def _lorenz96_rhs(y: np.ndarray, F: float) -> np.ndarray:
    K = y.size
    yp1 = np.roll(y, -1)
    ym1 = np.roll(y, 1)
    ym2 = np.roll(y, 2)
    return (yp1 - ym2) * ym1 - y + F

def lorenz96_generator(theta: np.ndarray, *, K: int = 8, F: float = 10.0,
                       dt: float = 3.0/40.0, T: float = 2.5,
                       y0: np.ndarray | None = None) -> np.ndarray:
    b0, b1, rho, sigma_e = map(float, theta[:4])
    steps = int(T / dt)
    y = np.random.randn(K) if y0 is None else np.array(y0, dtype=float).copy()
    r_prev = np.random.randn() * sigma_e
    traj = np.zeros((steps, K), dtype=float)

    for n in range(steps):
        g_prev = b0 + b1 * y + rho * r_prev + sigma_e * np.sqrt(max(1.0 - rho**2, 0.0)) * np.random.randn()
        def f(y_):
            return _lorenz96_rhs(y_, F) - g_prev
        k1 = f(y)
        k2 = f(y + 0.5 * dt * k1)
        k3 = f(y + 0.5 * dt * k2)
        k4 = f(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        # Clip values to prevent overflow/NaNs
        y = np.clip(y, -1e3, 1e3)
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            y = np.nan_to_num(y, nan=0.0, posinf=1e3, neginf=-1e3)
        r_prev = rho * r_prev + sigma_e * np.sqrt(max(1.0 - rho**2, 0.0)) * np.random.randn()
        traj[n] = y
    return traj.reshape(-1)
