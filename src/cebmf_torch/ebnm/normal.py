"""EBNM solver for the prior g = N(0, tau^2), fitted by marginal ML."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi

import torch
from torch import Tensor


@dataclass
class EBNormalResult:
    """Result of fitting the single-Gaussian EBNM prior g = N(0, tau^2).

    Attributes
    ----------
    post_mean : Tensor of shape (N,)
        Posterior means E[mu_i | betahat_i].
    post_mean2 : Tensor of shape (N,)
        Posterior second moments E[mu_i^2 | betahat_i].
    post_sd : Tensor of shape (N,)
        Posterior standard deviations sqrt(Var[mu_i | betahat_i]).
    tau2 : float
        Fitted prior variance, clamped to ``tau2_min``.
    log_lik : float
        Marginal log-likelihood at the fit.
    pi_slab : float
        Always 1.0; included for interface parity with point-prior results.
    """

    post_mean: Tensor
    post_mean2: Tensor
    post_sd: Tensor
    tau2: float
    log_lik: float
    pi_slab: float = 1.0


def _fit_tau2_heteroskedastic(
    beta_sq: Tensor,
    s_sq: Tensor,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Fit tau^2 by bisection on the marginal-likelihood score.

    Score(tau^2) = 0.5 * sum_i (beta_i^2 - (tau^2 + s_i^2)) / (tau^2 + s_i^2)^2.

    The marginal log-likelihood is unimodal under standard regularity (it is
    concave at the optimum because the Fisher information is positive there),
    so the score has at most one root on [0, infinity). The score is **not**
    globally monotone in tau^2: the per-observation second derivative
    ((tau^2 + s_i^2) - 2*beta_i^2) / (tau^2 + s_i^2)^3 is positive for those i
    with beta_i^2 large relative to tau^2 + s_i^2. What guarantees bisection
    works is the bracket: when the unconstrained ML is positive, score(0) > 0
    and score(hi) becomes negative for hi large enough (we double until that
    happens). Bisection on a sign-changing bracket of a continuous function
    converges by the intermediate value theorem; we do not need score
    monotonicity.

    Parameters
    ----------
    beta_sq : Tensor of shape (N,)
        Squared observations beta_i^2.
    s_sq : Tensor of shape (N,)
        Squared standard errors s_i^2.
    tol : float, default 1e-10
        Convergence tolerance on the score and the bracket width.
    max_iter : int, default 100
        Maximum number of bisection iterations.

    Returns
    -------
    float
        Estimated tau^2 in [0, +inf). Returns 0.0 when the unconstrained
        maximum-likelihood estimator is at the boundary.
    """

    def score(tau2: float) -> float:
        v = tau2 + s_sq
        return float(((beta_sq - v) / v.pow(2)).sum().item())

    # If score(0) <= 0, the unconstrained ML is at the boundary tau^2 = 0.
    if score(0.0) <= 0.0:
        return 0.0

    # Upper bound: 2 * max(beta^2) is safely above the unconstrained ML.
    lo, hi = 0.0, max(2.0 * float(beta_sq.max().item()), 1.0)
    while score(hi) > 0.0 and hi < 1e30:
        hi *= 2.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s = score(mid)
        if abs(s) < tol or (hi - lo) < tol * max(1.0, mid):
            return mid
        if s > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ebnm_normal(
    betahat: Tensor,
    sebetahat: Tensor | None = None,
    tau2_min: float = 0.0,
) -> EBNormalResult:
    """Fit g = N(0, tau^2) by marginal maximum likelihood.

    The marginal model is betahat_i ~ N(0, tau^2 + s_i^2). When all s_i are
    equal (or zero), the score equation has the closed form
    tau^2_hat = max(0, mean(betahat^2 - s^2)). When s_i are heterogeneous,
    the score is non-linear in tau^2 and is solved numerically by bisection
    on [0, +inf) (see :func:`_fit_tau2_heteroskedastic`).

    Parameters
    ----------
    betahat : Tensor of shape (N,)
        Observations.
    sebetahat : Tensor of shape (N,) or None, optional
        Standard errors. If ``None`` or identically zero, ``betahat`` is
        treated as exact. This is the case when using ``ebnm_normal`` as a
        Level-2 prior on regression coefficients.
    tau2_min : float, default 0.0
        Lower bound on the returned tau^2. Default 0 preserves standard ML
        semantics (returns the unconstrained ML estimator, which can be 0
        at the boundary). Hierarchical-prior callers should pass a small
        positive value (e.g. 1e-6) to avoid the degenerate sink at
        tau^2 = 0.

    Returns
    -------
    EBNormalResult
        Fitted tau^2 (heteroskedastic ML, clamped to ``tau2_min``), marginal
        log-likelihood at the fit, and posterior summaries from the standard
        Normal-Normal shrinkage formulas.

    Notes
    -----
    Invariants:

    - ``result.tau2 >= tau2_min``.
    - ``result.log_lik`` is finite for all inputs; the inner
      ``var_marg = max(tau2, 1e-30)`` floors the log-likelihood evaluation
      against the boundary tau^2 = 0.
    """
    betahat = betahat.detach()
    if sebetahat is None or torch.all(sebetahat == 0):
        # Zero-SE case: closed-form ML estimator.
        tau2 = float(betahat.pow(2).mean().item())
        tau2 = max(tau2, tau2_min)
        post_mean = betahat.clone()
        post_mean2 = betahat.pow(2)
        post_sd = torch.zeros_like(betahat)
        var_marg = torch.full_like(betahat, max(tau2, 1e-30))
    else:
        sebetahat = sebetahat.detach()
        s_sq = sebetahat.pow(2)
        beta_sq = betahat.pow(2)
        # Heteroskedastic ML by bisection on the score.
        tau2 = _fit_tau2_heteroskedastic(beta_sq, s_sq)
        tau2 = max(tau2, tau2_min)
        var_marg = tau2 + s_sq
        shrink = tau2 / var_marg
        post_mean = shrink * betahat
        post_var = shrink * s_sq
        post_mean2 = post_mean.pow(2) + post_var
        post_sd = post_var.sqrt()

    log_lik = float((-0.5 * betahat.pow(2) / var_marg - 0.5 * var_marg.log() - 0.5 * log(2 * pi)).sum().item())

    return EBNormalResult(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        tau2=tau2,
        log_lik=log_lik,
    )
