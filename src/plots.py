
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence

def plot_theta_mean_trajectories(theta_means: np.ndarray, true_theta: np.ndarray | None, outpath: str, title: str):
    T, D = theta_means.shape
    fig = plt.figure(figsize=(6, 4))
    for d in range(D):
        plt.plot(np.arange(T), theta_means[:, d], label=f"theta[{d}]")
    if true_theta is not None:
        for d in range(len(true_theta)):
            plt.axhline(true_theta[d], linestyle="--")
    plt.xlabel("iteration")
    plt.ylabel("ensemble mean")
    plt.title(title)
    plt.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_mmd2_curve(mmd2: np.ndarray, outpath: str, title: str):
    fig = plt.figure(figsize=(6, 4))
    plt.plot(np.arange(len(mmd2)), mmd2)
    plt.xlabel("iteration")
    plt.ylabel("MMD^2 (mean across ensemble)")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_histograms_per_dim(samples: np.ndarray, true_theta: np.ndarray | None, outpath_prefix: str):
    M, D = samples.shape
    for d in range(D):
        fig = plt.figure(figsize=(5, 3.2))
        plt.hist(samples[:, d], bins=30, density=True)
        if true_theta is not None:
            plt.axvline(true_theta[d], linestyle="--")
        plt.xlabel(f"theta[{d}]")
        plt.ylabel("density")
        plt.title(f"Posterior samples for dim {d}")
        fig.tight_layout()
        fig.savefig(f"{outpath_prefix}_dim{d}.png", dpi=150)
        plt.close(fig)

def plot_contamination_curve(contams: Sequence[float], post_means: Sequence[float], true_value: float, outpath: str, title: str):
    import numpy as np
    x = np.asarray(contams, dtype=float)
    y = np.asarray(post_means, dtype=float)
    fig = plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o")
    plt.axhline(true_value, linestyle="--")
    plt.xlabel("contamination fraction")
    plt.ylabel("posterior mean")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_contamination_rmse(contams: Sequence[float], rmses: Sequence[float], outpath: str, title: str):
    import numpy as np
    x = np.asarray(contams, dtype=float)
    y = np.asarray(rmses, dtype=float)
    fig = plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o")
    plt.xlabel("contamination fraction")
    plt.ylabel("RMSE")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
