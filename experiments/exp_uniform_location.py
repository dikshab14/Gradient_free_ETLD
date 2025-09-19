
import numpy as np
import csv
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gfetld_mmd import GFETLD, GFETLDConfig, GaussianPrior, uniform_location_generator
#from gfetld_mmd.plots import plot_contamination_curve, plot_contamination_rmse, plot_histograms_per_dim

OUTDIR = Path("./outputs/uniform_location")

def make_data(theta_true=1.0, N=100, contamination=0.0, z=10.0, seed=0):
    rng = np.random.default_rng(seed)
    n_clean = int(round((1.0 - contamination) * N))
    n_out = N - n_clean
    y_clean = rng.uniform(theta_true - 1.0, theta_true + 1.0, size=n_clean)
    y_out = z + rng.standard_normal(n_out)
    y = np.concatenate([y_clean, y_out]).reshape(-1, 1)
    rng.shuffle(y, axis=0)
    return y

def simulator(theta_vec: np.ndarray, J: int) -> np.ndarray:
    theta = float(theta_vec[0])
    return uniform_location_generator(theta, J)

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    contams = [0.0, 0.1, 0.2, 0.3, 0.4]
    post_means = []
    rmses = []
    for contam in contams:
        Y = make_data(theta_true=1.0, N=100, contamination=contam, z=10.0, seed=7)
        prior = GaussianPrior(mu=[1.0], var=[4.0])
        M = 10
        theta0 = prior.sample(M)
        cfg = GFETLDConfig(
            step_size=1e-3,
            n_steps=100,
            ensemble_size=M,
            n_sim_per_particle=50,
            kernel_bandwidth=5.0,
            temperature=10.0
        )
        sampler = GFETLD(dim_param=1, prior_loggrad=prior.logpdf_grad, simulator=simulator, obs_samples=Y, config=cfg)
        hist = sampler.run(theta0)
        theta_final = hist["theta_final"]
        post_mean = theta_final.mean(axis=0)[0]
        rmse = float(np.sqrt((post_mean - 1.0)**2))
        post_means.append(post_mean)
        rmses.append(rmse)

    #     plot_histograms_per_dim(theta_final, np.array([1.0]), str(OUTDIR / f"posterior_hist_contam{contam}"))

    # plot_contamination_curve(contams, post_means, true_value=1.0, outpath=str(OUTDIR / "posterior_mean_vs_contamination.png"), title="Uniform location: posterior mean vs contamination")
    # plot_contamination_rmse(contams, rmses, outpath=str(OUTDIR / "rmse_vs_contamination.png"), title="Uniform location: RMSE vs contamination")

    with open(OUTDIR / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contamination", "posterior_mean", "rmse"])
        for c, m, r in zip(contams, post_means, rmses):
            w.writerow([c, m, r])
    print(f"Saved outputs to: {OUTDIR.resolve()}")

if __name__ == "__main__":
    main()
