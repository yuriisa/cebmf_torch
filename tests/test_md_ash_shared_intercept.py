"""Shared sign intercept: restrictions, exact predictions, and state compatibility."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from cebmf_torch.cebnm.md_ash import MdAshNet, md_ash_posterior_means, predict_md_ash


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def sample():
    g = torch.Generator().manual_seed(583)
    x = torch.randn(150, 1, generator=g, dtype=torch.float64)
    y = 0.8 + x[:, 0] + torch.randn(150, generator=g, dtype=torch.float64)
    return x, y, torch.ones_like(y)


def test_initial_shared_logit_uses_total_slab_mass():
    pi = torch.tensor([0.2, 0.1, 0.3, 0.3, 0.1], dtype=torch.float64)
    model = MdAshNet((), (), pi.log(), sign_intercept="shared")
    assert model.sign_intercepts.numel() == 1
    torch.testing.assert_close(model.sign_intercepts, torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(
        model.log_probs(torch.empty(1, 0, dtype=torch.float64)).exp(),
        torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]], dtype=torch.float64),
    )


@pytest.mark.parametrize("with_covariates", [False, True])
def test_shared_fit_replay_and_equal_component_model(with_covariates):
    x, y, se = sample()
    x = x if with_covariates else None
    fit = md_ash_posterior_means(x, y, se, scale=[0, 0.4, 1.5], sign_intercept="shared", covariate_roles="both")
    state = fit.model_param
    assert state["sign_intercept"] == "shared"
    assert state["state_dict"]["sign_intercepts"].shape == (1,)
    assert state["optimization"]["converged"]
    with patch("cebmf_torch.cebnm.md_ash._train_model", side_effect=AssertionError("refit")):
        replay = md_ash_posterior_means(x, y, se, model_param=state, max_iter=0)
    torch.testing.assert_close(fit.post_mean, replay.post_mean, rtol=0, atol=0)
    continued = md_ash_posterior_means(x, y, se, model_param=state, max_iter=5)
    assert continued.model_param["sign_intercept"] == "shared"
    prior = predict_md_ash(x, observed=torch.zeros(len(y), dtype=torch.bool), model_param=state)
    pi = prior.log_pi.exp()
    shares = pi[:, 2::2] / (pi[:, 1::2] + pi[:, 2::2])
    torch.testing.assert_close(shares[:, 0], shares[:, 1])
    if not with_covariates:
        assert prior.post_mean[0] > 0.1
    full = deepcopy(state)
    full["sign_intercept"] = "component"
    if with_covariates:
        full["state_dict"]["sign_intercepts"] = full["state_dict"]["sign_intercepts"].expand(2).clone()
    else:
        full["kind"] = "exchangeable"
        full["pi"] = pi[0]
        del full["state_dict"]
    args = {"betahat": y, "sebetahat": se, "target": -y / 3, "target_se": se * 1.2}
    a = predict_md_ash(x, model_param=state, **args)
    b = predict_md_ash(x, model_param=full, **args)
    for key in ("post_mean", "post_sd", "posterior_weights", "log_predictive", "log_pi"):
        torch.testing.assert_close(getattr(a, key), getattr(b, key), rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="differs from the saved fit"):
        md_ash_posterior_means(x, y, se, model_param=state, sign_intercept="component", max_iter=0)


def test_fresh_default_is_shared_and_old_states_remain_component():
    x, y, se = sample()
    fit = md_ash_posterior_means(x, y, se, scale=[0, 0.5, 2])
    assert fit.model_param["sign_intercept"] == "shared"
    fit = md_ash_posterior_means(x, y, se, scale=[0, 0.5, 2], sign_intercept="component")
    state = deepcopy(fit.model_param)
    assert state.pop("sign_intercept") == "component"
    replay = md_ash_posterior_means(x, y, se, model_param=state, max_iter=0)
    torch.testing.assert_close(fit.post_mean, replay.post_mean, rtol=0, atol=0)
    with pytest.raises(ValueError, match="sign_intercept"):
        md_ash_posterior_means(x, y, se, sign_intercept="invalid")


def test_shared_spike_only():
    x, y, se = sample()
    fit = md_ash_posterior_means(x, y, se, scale=[0], sign_intercept="shared")
    assert fit.model_param["kind"] == "exchangeable"
    torch.testing.assert_close(fit.post_mean, torch.zeros_like(y))
    replay = md_ash_posterior_means(x, y, se, model_param=fit.model_param, max_iter=0)
    assert replay.model_param["sign_intercept"] == "shared"


def test_zero_shared_intercept_is_polcash_model():
    from cebmf_torch.cebnm.lcash import PropOddsLcashNet

    q = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    pi = torch.cat((q[:1], (q[1:] / 2).repeat_interleave(2)))
    md = MdAshNet((0,), (), pi.log(), sign_intercept="shared")
    po = PropOddsLcashNet(1, len(q), log_pi_init=q.log())
    with torch.no_grad():
        md.magnitude.w.fill_(0.7)
        po.w.fill_(0.7)
    x = torch.linspace(-3, 3, 15, dtype=torch.float64)[:, None]
    weights = md.log_probs(x).exp()
    expected = po.log_probs(x).exp()
    torch.testing.assert_close(weights[:, 0], expected[:, 0])
    torch.testing.assert_close(weights[:, 1::2], expected[:, 1:] / 2)
    torch.testing.assert_close(weights[:, 2::2], expected[:, 1:] / 2)
