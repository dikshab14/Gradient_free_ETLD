
import numpy as np
import csv
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
from gfetld_mmd import GFETLD, GFETLDConfig, GaussianPrior, gaussian_location_generator

OUTDIR = Path("./outputs/gaussian_location")
OUTDIR.mkdir(parents=True, exist_ok=True)

def make_data(theta_true=0.0, N=150, contamination=0.0, z=10.0, seed=0):
    rng = np.random.default_rng(seed)
    n_clean = int(round((1.0 - contamination) * N))
    n_out = N - n_clean
    y_clean = theta_true + rng.standard_normal(n_clean)
    y_out = z + rng.standard_normal(n_out)
    y = np.concatenate([y_clean, y_out]).reshape(-1, 1)
    rng.shuffle(y, axis=0)
    return y

def simulator(theta_vec: np.ndarray, J: int) -> np.ndarray:
    theta = float(theta_vec[0])
    return gaussian_location_generator(theta, J)

def run_once(contam: float, seed: int) -> float:
    # Data
    Y = make_data(theta_true=0.0, N=150, contamination=contam, z=10.0, seed=seed)
    # Prior & init
    prior = GaussianPrior(mu=[0.5], var=[1.0])
    M = 10
    theta0 = prior.sample(M)
    # Sampler config
    cfg = GFETLDConfig(
        step_size=1e-3,
        n_steps=100,
        ensemble_size=M,
        n_sim_per_particle=50,
        kernel_bandwidth=1.0,
        temperature=30.0
    )
    sampler = GFETLD(dim_param=1, prior_loggrad=prior.logpdf_grad, simulator=simulator, obs_samples=Y, config=cfg)
    hist = sampler.run(theta0)
    theta_final = hist["theta_final"]
    post_mean = float(theta_final.mean(axis=0)[0])
    rmse = abs(post_mean - 0.0)
    return rmse

def main():
    contams = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    seeds = list(range(10))  # average over 10 seeds
    rmse_mean = []
    rmse_std = []
    all_rows = []
    for c in contams:
        vals = [run_once(c, s) for s in seeds]
        rmse_mean.append(float(np.mean(vals)))
        rmse_std.append(float(np.std(vals)))
        for s, v in zip(seeds, vals):
            all_rows.append([c, s, v])
        print(f"contam={c:.2f}  RMSE mean={rmse_mean[-1]:.4f}  std={rmse_std[-1]:.4f}")

    # Save CSV (per-seed values and summary)
    # with open(OUTDIR / "gaussian_rmse_per_seed.csv", "w", newline="") as f:
    #     w = csv.writer(f); w.writerow(["contamination", "seed", "rmse"])
    #     w.writerows(all_rows)
    with open(OUTDIR / "gaussian_rmse_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["contamination", "rmse_mean", "rmse_std"])
        for c, m, s in zip(contams, rmse_mean, rmse_std):
            w.writerow([c, m, s])

    # # Plot RMSE vs contamination with error bars
    # fig = plt.figure(figsize=(6, 4))
    # x = np.array(contams, dtype=float)
    # y = np.array(rmse_mean, dtype=float)
    # yerr = np.array(rmse_std, dtype=float)
    # plt.errorbar(x, y, yerr=yerr, fmt="o-")
    # plt.xlabel("outlier proportion")
    # plt.ylabel("RMSE of posterior mean")
    # plt.title("Gaussian location (misspecified): RMSE vs outlier proportion")
    # fig.tight_layout()
    # fig.savefig(OUTDIR / "gaussian_rmse_vs_contamination.png", dpi=150)
    # plt.close(fig)
    

if __name__ == "__main__":
    main()
