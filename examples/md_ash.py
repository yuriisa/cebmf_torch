"""Fit MD-ASH to synthetic gene effects and predict every supplied gene.

Run from the repository root after installing the package:
    python examples/md_ash.py --output md-ash-example --plot

The optional plot requires matplotlib. These data teach the input contract.
They are not a benchmark or a model of a specific trait.
"""

import argparse
import csv
import json
from pathlib import Path

# BEGIN public_imports
import torch

from cebmf_torch import cEBMF
from cebmf_torch.cebnm import md_ash_posterior_means, predict_md_ash

# END public_imports


def run(output, plot=False):
    # BEGIN setup
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(125)
    dtype = torch.float64
    n = 1000
    # END setup

    # BEGIN observations
    # Generate a true LoF effect and separate source and target observations.
    s_het = torch.rand(n, generator=rng, dtype=dtype).pow(4)
    nonzero = torch.rand(n, generator=rng, dtype=dtype) < 0.15
    beta = nonzero * (0.1 + 0.5 * s_het.sqrt()) * torch.randn(n, generator=rng, dtype=dtype)
    sebetahat = 0.08 + 0.12 * torch.rand(n, generator=rng, dtype=dtype)
    betahat = beta + sebetahat * torch.randn(n, generator=rng, dtype=dtype)
    target_se = torch.full((n,), 0.1, dtype=dtype)
    target = beta + target_se * torch.randn(n, generator=rng, dtype=dtype)
    source_mac = torch.randint(0, 25, (n,), generator=rng).to(dtype)
    target_mac = torch.randint(0, 25, (n,), generator=rng).to(dtype)
    source_mac[::19] = float("nan")
    target_mac[::17] = float("nan")
    betahat[::7] = float("nan")
    target[::11] = float("nan")

    # Unknown counts pass. Source and target filters act independently.
    observed = torch.isfinite(betahat) & torch.isfinite(sebetahat) & (sebetahat > 0)
    observed &= torch.isnan(source_mac) | (source_mac >= 6)
    scored = torch.isfinite(target) & torch.isfinite(target_se) & (target_se > 0)
    scored &= torch.isnan(target_mac) | (target_mac >= 6)
    # END observations

    # BEGIN covariates
    # Missense evidence is correlated with the true LoF effect in this example.
    Z_mis = 0.6 * beta / 0.08 + torch.randn(n, generator=rng, dtype=dtype)
    s_het[::9] = float("nan")
    Z_mis[::5] = float("nan")
    terms = torch.stack((s_het, Z_mis.abs(), Z_mis), dim=1)
    missing = ~torch.isfinite(terms)
    imputation_mean = torch.nanmean(terms[observed], dim=0)
    if not torch.isfinite(imputation_mean).all():
        raise ValueError("Each term needs at least one finite value among fitting rows.")
    completed = torch.where(missing, imputation_mean, terms)
    # One missense indicator receives both roles, hence two separate slopes.
    X = torch.cat((completed, missing[:, [0, 2]].to(dtype)), dim=1)
    names = ["s_het", "abs(Z_mis)", "Z_mis", "s_het_missing", "missense_missing"]
    roles = ["magnitude", "magnitude", "direction", "magnitude", "both"]
    # END covariates

    # BEGIN fit_predict
    fit = md_ash_posterior_means(
        X[observed],
        betahat[observed],
        sebetahat[observed],
        covariate_roles=roles,
        sign_intercept="shared",
    )
    prediction = predict_md_ash(
        X,
        betahat,
        sebetahat,
        observed=observed,
        model_param=fit.model_param,
    )
    # Rows without usable source data receive their conditional prior.
    prior = predict_md_ash(X[~observed], model_param=fit.model_param)
    torch.testing.assert_close(prediction.post_mean[~observed], prior.post_mean)
    torch.testing.assert_close(prediction.posterior_weights[~observed], prior.posterior_weights)
    torch.testing.assert_close(
        prediction.prob_negative + prediction.prob_zero + prediction.prob_positive,
        torch.ones(n, dtype=dtype),
    )
    # END fit_predict

    # BEGIN score
    score = predict_md_ash(
        X[scored],
        betahat[scored],
        sebetahat[scored],
        observed=observed[scored],
        target=target[scored],
        target_se=target_se[scored],
        model_param=fit.model_param,
    )
    mean_lpd = score.log_predictive.mean().item()
    # END score

    # A no-covariate reference uses the same component standard deviations.
    baseline = md_ash_posterior_means(
        None,
        betahat[observed],
        sebetahat[observed],
        scale=fit.model_param["scale"],
        sign_intercept="shared",
    )
    reference = predict_md_ash(None, betahat, sebetahat, observed=observed, model_param=baseline.model_param)
    baseline_score = predict_md_ash(
        None,
        betahat[scored],
        sebetahat[scored],
        observed=observed[scored],
        target=target[scored],
        target_se=target_se[scored],
        model_param=baseline.model_param,
    )

    # BEGIN matrix
    # A separate small matrix example uses known observation SE 0.2.
    loading = torch.linspace(-1, 1, 40, dtype=dtype)
    factor = torch.linspace(-1, 1, 12, dtype=dtype)
    Y = loading[:, None] * factor[None, :] + 0.2 * torch.randn(40, 12, generator=rng, dtype=dtype)
    Y[0, :] = float("nan")
    model = cEBMF(
        Y,
        K=1,
        S=0.2,
        X_l=loading[:, None],
        prior_L="md_ash",
        prior_L_kwargs={"covariate_roles": "both"},
        device="cpu",
    )
    model.initialise_factors()  # Default SVD initialization.
    matrix_fit = model.fit(maxit=4)
    assert torch.isfinite(matrix_fit.L).all() and torch.isfinite(matrix_fit.F).all()
    # END matrix

    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_param": fit.model_param,
            "imputation_mean": imputation_mean,
            "covariate_names": names,
            "covariate_roles": roles,
        },
        output / "fit.pt",
    )
    summary = {
        "prediction_rows": n,
        "fitting_rows": observed.sum().item(),
        "prior_only_rows": (~observed).sum().item(),
        "scored_rows": scored.sum().item(),
        "scored_prior_only_rows": (scored & ~observed).sum().item(),
        "mean_lpd": mean_lpd,
        "reference_mean_lpd": baseline_score.log_predictive.mean().item(),
        "mean_paired_lpd_difference": (score.log_predictive - baseline_score.log_predictive).mean().item(),
        "optimization": fit.model_param["optimization"],
        "reference_optimization": baseline.model_param["optimization"],
        "matrix_loading_shape": list(matrix_fit.L.shape),
        "matrix_factor_shape": list(matrix_fit.F.shape),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "gene",
                "true_effect",
                "source_used",
                "target_scored",
                "post_mean",
                "post_sd",
                "prob_negative",
                "prob_zero",
                "prob_positive",
                "reference_post_mean",
            ]
        )
        writer.writerows(
            zip(
                (f"gene_{i:04d}" for i in range(n)),
                beta.tolist(),
                observed.tolist(),
                scored.tolist(),
                prediction.post_mean.tolist(),
                prediction.post_sd.tolist(),
                prediction.prob_negative.tolist(),
                prediction.prob_zero.tolist(),
                prediction.prob_positive.tolist(),
                reference.post_mean.tolist(),
                strict=True,
            )
        )
    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
        for ax, mask, title in zip(
            axes, (observed, ~observed), ("Usable source LoF", "Missing or MAC < 6 source LoF"), strict=True
        ):
            ax.scatter(beta[mask], reference.post_mean[mask], s=9, alpha=0.45, label="No covariates", color="#8c8c8c")
            ax.scatter(
                beta[mask],
                prediction.post_mean[mask],
                s=9,
                alpha=0.55,
                label="s_het, |Z_mis|, Z_mis\n+ missing indicators",
                color="#007a87",
            )
            ax.plot([-1.5, 1.5], [-1.5, 1.5], "k--", lw=0.7, label="Posterior mean = true effect")
            ax.set(title=f"{title} (n={int(mask.sum())})", xlabel="True synthetic LoF effect")
            ax.grid(alpha=0.15)
        axes[0].set_ylabel("Posterior mean LoF effect")
        axes[1].legend(fontsize=8, loc="upper left")
        fig.suptitle(
            "Shared MD-ASH: posterior means for all 1,000 supplied genes\n"
            "One synthetic example. Target data never enter fitting."
        )
        fig.tight_layout()
        fig.savefig(output / "posterior_means.png", dpi=160)
        plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if "optimization" not in k}, indent=2))
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("md-ash-example"))
    parser.add_argument("--plot", action="store_true", help="Write a posterior-mean plot using matplotlib.")
    arguments = parser.parse_args()
    run(arguments.output, arguments.plot)
