
import numpy as np
from pathlib import Path
import csv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gfetld_mmd import GFETLD, GFETLDConfig, GaussianPrior, lorenz96_generator
from gfetld_mmd.plots import plot_theta_mean_trajectories, plot_mmd2_curve, plot_histograms_per_dim

OUTDIR = Path("./outputs/lorenz96")
TRUE_THETA = np.array([2.0, 0.8, 0.9, 1.7])

def make_observed(N_traj=20, seed=0):
    rng = np.random.default_rng(seed)
    Y = []
    for i in range(N_traj):
        y0 = rng.standard_normal(8)
        traj = lorenz96_generator(TRUE_THETA, K=8, F=10.0, dt=3.0/40.0, T=2.5, y0=y0)
        Y.append(traj.reshape(-1))  # Flatten each trajectory
    return np.stack(Y, axis=0)

def simulator(theta_vec: np.ndarray, J: int) -> np.ndarray:
    outs = []
    for j in range(int(J)):
        traj = lorenz96_generator(theta_vec, K=8, F=10.0, dt=3.0/40.0, T=2.5, y0=None)
        outs.append(traj)
    return np.stack(outs, axis=0)

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    Y = make_observed(N_traj=20, seed=123)
    prior = GaussianPrior(mu=[2.0, 1.0, 1.0, 2.0], var=[1.0, 1.0, 2.0, 1.0])
    M = 50
    theta0 = prior.sample(M)
    cfg = GFETLDConfig(
        step_size=1e-3,
        n_steps=10,
        ensemble_size=M,
        n_sim_per_particle=80,
        kernel_bandwidth=1.0,
        temperature=20.0
    )
    sampler = GFETLD(dim_param=4, prior_loggrad=prior.logpdf_grad, simulator=simulator, obs_samples=Y, config=cfg)
    hist = sampler.run(theta0)

    theta_mean = hist["theta_mean"]
    mmd2 = hist["mmd2_mean"]
    theta_final = hist["theta_final"]
    post_mean = theta_final.mean(axis=0)

    # plot_theta_mean_trajectories(theta_mean, TRUE_THETA, str(OUTDIR / "theta_mean_trajectories.png"), title="Lorenz96: ensemble mean over iterations")
    # plot_mmd2_curve(mmd2, str(OUTDIR / "mmd2_curve.png"), title="Lorenz96: mean MMD^2 over iterations")
    # plot_histograms_per_dim(theta_final, TRUE_THETA, str(OUTDIR / "posterior_hist"))

    with open(OUTDIR / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param_index", "posterior_mean", "true_value", "abs_error"])
        for i in range(4):
            w.writerow([i, post_mean[i], TRUE_THETA[i], abs(post_mean[i]-TRUE_THETA[i])])
    print(f"Saved outputs to: {OUTDIR.resolve()}")

if __name__ == "__main__":
    main()
