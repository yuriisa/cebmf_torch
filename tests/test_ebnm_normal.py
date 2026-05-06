"""Tests for the single-Gaussian EBNM solver ``ebnm_normal``.

Specifications are in Section 9.4 of
``cebmf_torch_hierarchical_priors_design.md``.
"""

import math

import torch

from cebmf_torch.ebnm import ebnm_normal
from cebmf_torch.ebnm.normal import _fit_tau2_heteroskedastic


def _marginal_log_lik(betahat: torch.Tensor, sebetahat: torch.Tensor, tau2: float) -> float:
    """Marginal log-likelihood under betahat_i ~ N(0, tau^2 + s_i^2)."""
    var_marg = tau2 + sebetahat.pow(2)
    var_marg = torch.clamp(var_marg, min=1e-30)
    return float((-0.5 * betahat.pow(2) / var_marg - 0.5 * var_marg.log() - 0.5 * math.log(2 * math.pi)).sum().item())


def _score(betahat: torch.Tensor, sebetahat: torch.Tensor, tau2: float) -> float:
    """Score d/d(tau^2) of the marginal log-likelihood (factor 1/2 dropped:
    only the sign and the location of the root matter for the test)."""
    v = tau2 + sebetahat.pow(2)
    return float(((betahat.pow(2) - v) / v.pow(2)).sum().item())


def test_ebnm_normal_zero_se():
    torch.manual_seed(0)
    n = 1000
    tau2_true = 4.0
    beta = torch.randn(n) * math.sqrt(tau2_true)

    res = ebnm_normal(beta, sebetahat=None)

    assert abs(res.tau2 - tau2_true) / tau2_true < 0.10
    assert math.isfinite(res.log_lik)
    # Zero-SE branch: posterior mean is the observation itself.
    assert torch.allclose(res.post_mean, beta)
    assert torch.allclose(res.post_sd, torch.zeros_like(beta))
    assert res.pi_slab == 1.0


def test_ebnm_normal_with_se_homoskedastic():
    torch.manual_seed(1)
    n = 5000
    tau2_true = 4.0
    s_val = 0.5
    s = torch.full((n,), s_val)
    mu = torch.randn(n) * math.sqrt(tau2_true)
    beta = mu + torch.randn(n) * s_val

    res = ebnm_normal(beta, sebetahat=s)

    # ML should recover tau^2 within 10%.
    assert abs(res.tau2 - tau2_true) / tau2_true < 0.10

    # In the homoskedastic case the score equation has closed form
    # tau^2_hat = max(0, mean(beta^2 - s^2)), which equals MoM. Both
    # estimators should coincide.
    mom = max(0.0, float((beta.pow(2) - s.pow(2)).mean().item()))
    assert abs(res.tau2 - mom) < 1e-6

    assert math.isfinite(res.log_lik)


def test_ebnm_normal_with_se_heteroskedastic():
    torch.manual_seed(2)
    n = 5000
    tau2_true = 4.0
    s = 0.1 + 0.9 * torch.rand(n)  # Uniform(0.1, 1.0)
    mu = torch.randn(n) * math.sqrt(tau2_true)
    beta = mu + torch.randn(n) * s

    res = ebnm_normal(beta, sebetahat=s)

    # Recover tau^2 within 10%.
    assert abs(res.tau2 - tau2_true) / tau2_true < 0.10

    # Crucially: ML must beat MoM in marginal log-likelihood.
    tau2_mom = max(0.0, float((beta.pow(2) - s.pow(2)).mean().item()))
    ll_ml = _marginal_log_lik(beta, s, res.tau2)
    ll_mom = _marginal_log_lik(beta, s, tau2_mom)
    assert ll_ml >= ll_mom
    # And strictly greater than MoM when MoM is interior (heteroskedastic
    # case has tau2_mom strictly different from the ML root in general).
    assert ll_ml - ll_mom > 0.0


def test_ebnm_normal_score_at_optimum():
    torch.manual_seed(3)
    n = 2000
    tau2_true = 4.0
    s = 0.1 + 0.9 * torch.rand(n)
    mu = torch.randn(n) * math.sqrt(tau2_true)
    beta = mu + torch.randn(n) * s

    res = ebnm_normal(beta, sebetahat=s)

    # Either at the boundary tau^2 = 0, or the score is approximately zero.
    if res.tau2 == 0.0:
        # Boundary: the unconstrained ML is at zero, i.e. score(0) <= 0.
        assert _score(beta, s, 0.0) <= 1e-8
    else:
        # Interior optimum: score should be near zero. Bisection runs with
        # tolerance 1e-10 on the un-scaled score, but float32 reductions over
        # n observations limit the achievable accuracy of the assertion-side
        # recomputation. Use a tolerance scaled by n.
        assert abs(_score(beta, s, res.tau2)) < 1e-3


def test_ebnm_normal_zero_signal():
    torch.manual_seed(4)
    n = 2000
    s = 0.1 + 0.9 * torch.rand(n)
    # No true signal: betahat = noise only.
    beta = torch.randn(n) * s

    res = ebnm_normal(beta, sebetahat=s)

    # tau^2 should be at the boundary 0 (or very close).
    assert res.tau2 == 0.0
    assert math.isfinite(res.log_lik)


def test_ebnm_normal_loglik_finite():
    torch.manual_seed(5)

    # Regime 1: zero-SE input.
    beta = torch.randn(500) * 2.0
    res = ebnm_normal(beta, sebetahat=None)
    assert math.isfinite(res.log_lik)

    # Regime 2: homoskedastic with signal.
    s = torch.full((500,), 0.3)
    mu = torch.randn(500) * 2.0
    beta = mu + torch.randn(500) * 0.3
    res = ebnm_normal(beta, sebetahat=s)
    assert math.isfinite(res.log_lik)

    # Regime 3: heteroskedastic with signal.
    s = 0.1 + 0.9 * torch.rand(500)
    mu = torch.randn(500) * 2.0
    beta = mu + torch.randn(500) * s
    res = ebnm_normal(beta, sebetahat=s)
    assert math.isfinite(res.log_lik)

    # Regime 4: zero signal (boundary tau^2 = 0).
    s = 0.1 + 0.9 * torch.rand(500)
    beta = torch.randn(500) * s
    res = ebnm_normal(beta, sebetahat=s)
    assert math.isfinite(res.log_lik)

    # Regime 5: pathological all-zero input with zero SE.
    beta = torch.zeros(500)
    res = ebnm_normal(beta, sebetahat=None)
    assert math.isfinite(res.log_lik)


def test_ebnm_normal_tau2_min_clamp():
    """``tau2_min`` is the single home of the clamp; the returned tau^2
    must not fall below it."""
    torch.manual_seed(6)
    n = 1000
    s = 0.1 + 0.9 * torch.rand(n)
    beta = torch.randn(n) * s  # zero signal -> unconstrained ML at 0

    res = ebnm_normal(beta, sebetahat=s, tau2_min=1e-3)
    assert res.tau2 >= 1e-3

    # Zero-SE branch also clamps.
    res = ebnm_normal(torch.zeros(100), sebetahat=None, tau2_min=1e-6)
    assert res.tau2 >= 1e-6


def test_fit_tau2_heteroskedastic_boundary():
    """``_fit_tau2_heteroskedastic`` returns 0 when score(0) <= 0.

    Construct an input where the heteroskedastic score at tau^2 = 0 is
    non-positive: shrink betahat below the noise scale s.
    """
    torch.manual_seed(7)
    n = 1000
    s = 0.1 + 0.9 * torch.rand(n)
    # Scale residuals to be much smaller than s, so beta^2 - s^2 < 0
    # at every i and the heteroskedastic score(0) is strictly negative.
    beta = 0.01 * torch.randn(n) * s
    tau2 = _fit_tau2_heteroskedastic(beta.pow(2), s.pow(2))
    assert tau2 == 0.0
