"""Independent references and exact-distribution identities for half-normal MD-ASH."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from cebmf_torch.cebnm.md_ash import md_ash_posterior_means, predict_md_ash
from cebmf_torch.utils.distribution_operation import get_data_loglik_normal_torch
from cebmf_torch.utils.halfnormal import get_data_loglik_halfnormal_torch, halfnormal_moments
from cebmf_torch.utils.posterior import posterior_mean_norm

REFERENCE = json.loads((Path(__file__).parent / "fixtures/halfnormal_r_reference.json").read_text())


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def fixed_state(scale, pi):
    return {
        "version": 1,
        "solver": "MD-ASH",
        "prior": "halfnormal",
        "kind": "exchangeable",
        "scale": tensor(scale),
        "pi": tensor(pi),
        "covariate_roles": (),
        "feature_mean": tensor([]),
        "feature_sd": tensor([]),
        "optimization": {"converged": None},
        "fit_options": {"penalty": 10.0, "coefficient_prior_scale": 1.0, "coefficient_prior_df": 5.0},
    }


@pytest.mark.parametrize("y,se", [(1, 1e-8), (1e8, 1)])
@pytest.mark.parametrize("scale,pi", [([0], [1]), ([0, 1], [1, 0, 0])])
def test_spike_predictive_density_is_independent_of_extreme_source(y, se, scale, pi):
    target, error = tensor([-2, 0, 3]), tensor([0.5, 1, 2])
    result = predict_md_ash(
        None, tensor([y] * 3), tensor([se] * 3), model_param=fixed_state(scale, pi), target=target, target_se=error
    )
    expected = torch.distributions.Normal(torch.zeros_like(error), error).log_prob(target)
    torch.testing.assert_close(result.log_predictive, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("sign", [-1, 1])
def test_conflicting_halfnormal_predictive_density_has_correct_zero_limit(sign):
    pi = [0, 1, 0] if sign == -1 else [0, 0, 1]
    target, error = tensor([-2, 0, 3]), tensor([0.5, 1, 2])
    result = predict_md_ash(
        None,
        tensor([-sign] * 3),
        tensor([1e-8] * 3),
        model_param=fixed_state([0, 1], pi),
        target=target,
        target_se=error,
    )
    # The truncated posterior concentrates at zero with mean approximately sign * 1e-16.
    expected = torch.distributions.Normal(torch.zeros_like(error), error).log_prob(target)
    torch.testing.assert_close(result.log_predictive, expected, rtol=1e-13, atol=1e-13)


@pytest.fixture(autouse=True)
def one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_fixed_mixture_matches_ashr():
    case = REFERENCE["fixed"]
    y, s, scale, pi = (tensor(case[k]) for k in ("x", "s", "scale", "pi"))
    likelihood = get_data_loglik_halfnormal_torch(y, s, scale)
    result = predict_md_ash(None, y, s, model_param=fixed_state(scale, pi))
    torch.testing.assert_close(torch.logsumexp(likelihood + pi.log(), 1).sum(), tensor(case["log_lik"]))
    torch.testing.assert_close(result.post_mean, tensor(case["mean"]), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(result.post_mean2, tensor(case["mean2"]), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("case", REFERENCE["fits"], ids=lambda c: f"penalty_{c['penalty']}")
def test_fitted_mixture_matches_ashr(case):
    y, s, scale = (tensor(case[k]) for k in ("x", "s", "scale"))
    result = md_ash_posterior_means(None, y, s, scale=scale, penalty=case["penalty"], sign_intercept="component")
    torch.testing.assert_close(
        result.loss + result.regularization_loss, tensor(case["objective"]), rtol=1e-7, atol=1e-6
    )
    torch.testing.assert_close(result.post_mean, tensor(case["mean"]), rtol=3e-4, atol=1e-4)
    torch.testing.assert_close(result.post_mean2, tensor(case["mean2"]), rtol=5e-4, atol=2e-4)


def test_symmetric_pairs_equal_normal_mixture():
    y, s, scale = tensor([-8, -0.4, 0, 1, 15]), tensor([0.3, 1, 2, 0.1, 3]), tensor([0, 0.5, 2])
    pi = tensor([0.4, 0.25, 0.35])
    signed_pi = torch.cat((pi[:1], pi[1:].repeat_interleave(2) / 2))
    signed = get_data_loglik_halfnormal_torch(y, s, scale)
    normal = get_data_loglik_normal_torch(y, s, location=torch.zeros_like(scale), scale=scale)
    collapsed = torch.cat((signed[:, :1], torch.logsumexp(signed[:, 1:].reshape(len(y), -1, 2), 2) - math.log(2)), 1)
    torch.testing.assert_close(collapsed, normal, rtol=1e-13, atol=1e-13)
    actual = predict_md_ash(None, y, s, model_param=fixed_state(scale, signed_pi))
    expected = posterior_mean_norm(y, s, pi.log(), data_loglik=normal, location=torch.zeros_like(scale), scale=scale)
    for key in ("post_mean", "post_mean2", "post_sd"):
        torch.testing.assert_close(getattr(actual, key), getattr(expected, key), rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("y,s,tau", [(-5, 0.2, 1), (-0.4, 0.3, 0.5), (0, 2, 0.25), (3, 0.4, 2)])
def test_components_against_independent_quadrature(y, s, tau):
    nodes, weights = np.polynomial.legendre.leggauss(500)
    theta = (nodes + 1) * 6 * tau
    prior = 2 * np.exp(-0.5 * (theta / tau) ** 2) / (tau * math.sqrt(2 * math.pi))
    likelihood = get_data_loglik_halfnormal_torch(tensor([y]), tensor([s]), tensor([0, tau]))
    mean, second, variance = halfnormal_moments(tensor([y]), tensor([s]), tensor([0, tau]), return_variance=True)
    for index, sign in enumerate((-1, 1)):
        value = sign * theta
        mass = weights * 6 * tau * prior * np.exp(-0.5 * ((y - value) / s) ** 2) / (s * math.sqrt(2 * math.pi))
        total = mass.sum()
        expected = (mass * value).sum() / total
        assert likelihood[0, index + 1].item() == pytest.approx(math.log(total), abs=1e-9)
        assert mean[0, index].item() == pytest.approx(expected, rel=1e-8)
        assert second[0, index].item() == pytest.approx((mass * value**2).sum() / total, rel=1e-8)
        assert variance[0, index].item() == pytest.approx((mass * (value - expected) ** 2).sum() / total, rel=1e-8)


def test_extreme_conflict_tiny_error_and_reflection():
    y, s = tensor([-1e4, 1]), tensor([1, 1e-8])
    state = fixed_state([0, 1], [0, 0, 1])
    result = predict_md_ash(None, y, s, model_param=state)
    assert result.post_mean[0].item() == pytest.approx(1e-4, rel=1e-6)
    assert result.post_sd[0].item() == pytest.approx(1e-4, rel=1e-6)
    assert result.post_sd[1].item() == pytest.approx(1e-8, rel=1e-6)
    other = predict_md_ash(None, -y, s, model_param=fixed_state([0, 1], [0, 1, 0]))
    torch.testing.assert_close(result.post_mean, -other.post_mean)
    torch.testing.assert_close(result.post_mean2, other.post_mean2)
    torch.testing.assert_close(result.prob_positive, other.prob_negative)


def test_predictive_identity_against_joint_integration_and_prior_only():
    x, s, z, t, tau = 0.4, 0.3, -0.2, 0.5, 0.7
    pi = tensor([0.2, 0.3, 0.5])
    scale = tensor([0, tau])
    result = predict_md_ash(
        None, tensor([x]), tensor([s]), model_param=fixed_state(scale, pi), target=tensor([z]), target_se=tensor([t])
    )
    nodes, weights = np.polynomial.legendre.leggauss(300)
    theta = (nodes + 1) * 6 * tau

    def density(value, center, error):
        return np.exp(-0.5 * ((value - center) / error) ** 2) / (error * math.sqrt(2 * math.pi))

    joint = pi[0].item() * density(x, 0, s) * density(z, 0, t)
    for index, sign in enumerate((-1, 1)):
        joint += (
            weights
            * 6
            * tau
            * pi[index + 1].item()
            * 2
            * density(theta, 0, tau)
            * density(x, sign * theta, s)
            * density(z, sign * theta, t)
        ).sum()
    marginal = torch.logsumexp(get_data_loglik_halfnormal_torch(tensor([x]), tensor([s]), scale) + pi.log(), 1)[0]
    assert result.log_predictive.item() == pytest.approx(math.log(joint) - marginal.item(), abs=1e-12)
    prior = predict_md_ash(None, model_param=fixed_state(scale, pi), target=tensor([z]), target_se=tensor([t]))
    expected = torch.logsumexp(get_data_loglik_halfnormal_torch(tensor([z]), tensor([t]), scale) + pi.log(), 1)
    torch.testing.assert_close(prior.log_predictive, expected)


def test_exact_spike_and_component_array_contract():
    y, s = tensor([-2, 0, 3]), tensor([0.1, 1, 2])
    fit = md_ash_posterior_means(None, y, s, scale=[0])
    result = predict_md_ash(None, y, s, model_param=fit.model_param, target=y, target_se=s)
    assert torch.equal(result.post_sd, torch.zeros_like(y))
    assert torch.equal(result.posterior_weights, torch.ones((3, 1), dtype=torch.float64))
    assert torch.equal(result.component_mean, torch.zeros((3, 1), dtype=torch.float64))
    assert torch.equal(result.component_sign, tensor([0]))
    state = fixed_state([0, 0.4, 2], [0.2, 0.1, 0.2, 0.3, 0.2])
    result = predict_md_ash(None, y, s, model_param=state, observed=torch.tensor([True, False, True]))
    assert result.component_sign.tolist() == [0, -1, 1, -1, 1]
    torch.testing.assert_close(result.component_sd[1], tensor([0, 0.4, 0.4, 2, 2]))
    torch.testing.assert_close(result.component_mean[1], torch.zeros(5, dtype=torch.float64))
    ratio = tensor([0.4, 2]) / torch.hypot(s[0], tensor([0.4, 2]))
    torch.testing.assert_close(result.component_mean[0, 1:], (ratio**2 * y[0]).repeat_interleave(2))
    torch.testing.assert_close(result.prob_zero + result.prob_negative + result.prob_positive, torch.ones_like(y))
