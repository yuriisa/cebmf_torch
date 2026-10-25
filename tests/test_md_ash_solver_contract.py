"""Input, state, penalty, and optimization contracts for the selected solver."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from cebmf_torch.cebnm.lcash import _train_model
from cebmf_torch.cebnm.md_ash import MdAshNet, _prior_weights, md_ash_posterior_means, predict_md_ash
from cebmf_torch.utils.halfnormal import get_data_loglik_halfnormal_torch
from cebmf_torch.utils.mixture import optimize_pi_logL_lbfgs


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def data():
    g = torch.Generator().manual_seed(31)
    X = torch.randn((80, 2), generator=g, dtype=torch.float64)
    y = X[:, 0] + torch.randn(80, generator=g, dtype=torch.float64)
    return X, y, torch.ones_like(y)


def fitted(roles="magnitude", **kwargs):
    X, y, se = data()
    return md_ash_posterior_means(
        X, y, se, covariate_roles=roles, scale=[0, 0.5, 2], sign_intercept="component", **kwargs
    )


@pytest.mark.parametrize(
    "roles, nm, nd", [("magnitude", 2, 0), ("direction", 0, 2), ("both", 2, 2), (["magnitude", "direction"], 1, 1)]
)
def test_roles_allocate_only_requested_slopes(roles, nm, nd):
    result = fitted(roles)
    params = result.model_param["state_dict"]
    assert params["magnitude.w"].numel() == nm
    assert params["direction_weight"].numel() == nd
    assert params["sign_intercepts"].numel() == 2
    assert result.model_param["optimization"]["initialization"]["conditional_uniform_mass"] == 0.01


def test_magnitude_role_is_default_and_negative_values_are_not_abs_transformed():
    X, y, se = data()
    a = md_ash_posterior_means(X, y, se, scale=[0, 0.5, 2], sign_intercept="component")
    b = fitted("magnitude")
    torch.testing.assert_close(a.post_mean, b.post_mean, rtol=0, atol=0)
    torch.testing.assert_close(a.model_param["feature_mean"], X.mean(0))
    assert a.model_param["state_dict"]["direction_weight"].numel() == 0


def test_prepared_indicator_is_standardized_and_constant_column_stays_inactive():
    X, y, se = data()
    X[:, 0] = (X[:, 0] > 0).double()
    X[:, 1] = 7
    result = md_ash_posterior_means(X, y, se, scale=[0, 0.5, 2], covariate_roles="both")
    state = result.model_param
    assert state["feature_sd"][0] > 0
    assert state["feature_sd"][1] == 0
    assert state["state_dict"]["magnitude.w"].numel() == 1
    torch.testing.assert_close(state["feature_mean"], X.mean(0))
    torch.testing.assert_close(state["feature_sd"][0], X[:, 0].std(unbiased=False))
    before = predict_md_ash(X, model_param=state)
    X[:, 1] = torch.arange(len(X)) * 1000.0
    after = predict_md_ash(X, model_param=state)
    torch.testing.assert_close(before.log_pi, after.log_pi, rtol=0, atol=0)


def test_saved_replay_and_predict_do_not_run_an_optimizer_or_change_state():
    X, y, se = data()
    result = fitted(["direction", "magnitude"], coefficient_prior_scale=0.7, penalty=3, coefficient_prior_df=4)
    state = deepcopy(result.model_param)
    with (
        patch("cebmf_torch.cebnm.md_ash._train_model", side_effect=AssertionError("refit")),
        patch("cebmf_torch.cebnm.md_ash.optimize_pi_logL_lbfgs", side_effect=AssertionError("refit")),
    ):
        replay = md_ash_posterior_means(X, y, se, model_param=state, max_iter=0)
        direct = predict_md_ash(X, y, se, model_param=state)
    torch.testing.assert_close(result.post_mean, replay.post_mean, rtol=0, atol=0)
    torch.testing.assert_close(result.post_mean, direct.post_mean, rtol=0, atol=0)
    assert replay.model_param["fit_options"] == state["fit_options"]
    assert replay.model_param["optimization"]["converged"] is None
    assert replay.model_param["optimization"]["previous_fit"] == state["optimization"]
    for key, value in state["state_dict"].items():
        torch.testing.assert_close(value, result.model_param["state_dict"][key], rtol=0, atol=0)
    replay.model_param["state_dict"]["direction_weight"].zero_()
    assert not torch.equal(
        replay.model_param["state_dict"]["direction_weight"], state["state_dict"]["direction_weight"]
    )


@pytest.mark.parametrize("mode", ["shared", "component"])
def test_replay_reports_current_objective_without_claiming_convergence(mode):
    X, y, se = data()
    fit = md_ash_posterior_means(X, y, se, covariate_roles="both", scale=[0, 0.5, 2], sign_intercept=mode)
    replay = md_ash_posterior_means(X, y + 10, se, model_param=fit.model_param, penalty=3, max_iter=0)
    diagnostics = replay.model_param["optimization"]
    assert diagnostics["status"] == "not_assessed"
    assert diagnostics["converged"] is None
    assert diagnostics["n_iter"] == diagnostics["max_iter"] == 0
    assert diagnostics["previous_fit"] == fit.model_param["optimization"]
    assert diagnostics["mean_objective"] == pytest.approx(((replay.loss + replay.regularization_loss) / len(y)).item())
    twice = md_ash_posterior_means(X, y, se, model_param=replay.model_param, max_iter=0)
    assert twice.model_param["optimization"]["previous_fit"] == fit.model_param["optimization"]


def test_batch_permutation_and_missing_source_contract():
    X, y, se = data()
    state = fitted("both").model_param
    observed = torch.arange(len(y)) % 3 != 0
    full = predict_md_ash(X, y, se, model_param=state, observed=observed)
    prior = predict_md_ash(X, model_param=state)
    torch.testing.assert_close(full.posterior_weights[~observed], prior.posterior_weights[~observed])
    y[~observed], se[~observed] = float("inf"), -1
    replaced = predict_md_ash(X, y, se, model_param=state, observed=observed)
    torch.testing.assert_close(full.post_mean, replaced.post_mean)
    order = torch.arange(len(y) - 1, -1, -1)
    reverse = predict_md_ash(X[order], y[order], se[order], model_param=state, observed=observed[order])
    torch.testing.assert_close(full.post_mean[order], reverse.post_mean)
    for start in (0, 40):
        batch = predict_md_ash(
            X[start : start + 40],
            y[start : start + 40],
            se[start : start + 40],
            model_param=state,
            observed=observed[start : start + 40],
        )
        torch.testing.assert_close(full.post_mean[start : start + 40], batch.post_mean)


def test_null_and_slope_penalties_match_native_objective():
    X, y, se = data()
    result = fitted("both")
    state = result.model_param
    pi, _ = _prior_weights(X, state)
    w = torch.cat((state["state_dict"]["magnitude.w"], state["state_dict"]["direction_weight"]))
    prior = torch.distributions.StudentT(torch.tensor(5.0, dtype=torch.float64), scale=1.0)
    expected = -9 * pi[:, 0].mean() - prior.log_prob(w).sum()
    torch.testing.assert_close(result.regularization_loss, expected)
    assert (result.loss + result.regularization_loss).item() / len(y) == pytest.approx(
        state["optimization"]["mean_objective"], abs=1e-12
    )
    no_slope = md_ash_posterior_means(X, y, se, model_param=state, max_iter=0, coefficient_prior_scale=None)
    torch.testing.assert_close(no_slope.regularization_loss, -9 * pi[:, 0].mean())
    assert no_slope.model_param["fit_options"]["coefficient_prior_scale"] is None


def test_interior_initialization_recovers_independent_signed_signal():
    # The feature precedes the observations; it is not computed from noisy y.
    x = torch.cat((-torch.ones(60), torch.ones(60))).double()[:, None]
    y = 0.6 * x[:, 0] + torch.randn(120, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    se = torch.ones_like(y)
    grid = [0, 0.3, 1, 3]
    likelihood = get_data_loglik_halfnormal_torch(y, se, grid)
    pi = optimize_pi_logL_lbfgs(likelihood, 10, zero_threshold=0, max_iter=2000)
    legacy = MdAshNet((), (0,), pi.clamp_min(1e-8).log(), sign_intercept="component")
    before = _train_model(legacy, x, likelihood, 500, 10, False, "MD-ASH", 1, 5)
    fit = md_ash_posterior_means(x, y, se, scale=grid, covariate_roles="direction", sign_intercept="component")
    after = fit.model_param["optimization"]
    assert before["converged"] and after["converged"]
    assert after["mean_objective"] < before["mean_objective"] - 0.02
    assert fit.model_param["state_dict"]["direction_weight"].item() > 1
    assert abs(legacy.direction_weight.item()) < 0.01
    predicted = predict_md_ash([[-1], [1]], model_param=fit.model_param)
    assert predicted.post_mean[0] < 0 < predicted.post_mean[1]


def test_finite_budget_reports_native_nonconvergence():
    with pytest.warns(RuntimeWarning, match="gradient tolerance"):
        result = fitted("both", max_iter=1)
    d = result.model_param["optimization"]
    assert not d["converged"]
    assert d["n_iter"] == 1
    assert d["max_iter"] == 1
    assert d["max_gradient"] > d["tolerance_grad"]


def test_exchangeable_saved_replay_and_single_spike_are_well_defined():
    _X, y, se = data()
    result = md_ash_posterior_means(None, y, se, scale=[0])
    assert result.model_param["kind"] == "exchangeable"
    assert torch.equal(result.post_mean, torch.zeros_like(y))
    assert torch.equal(result.post_sd, torch.zeros_like(y))
    replay = md_ash_posterior_means(None, y, se, model_param=result.model_param, max_iter=0)
    torch.testing.assert_close(replay.post_mean, result.post_mean)


@pytest.mark.parametrize("mode", ["shared", "component"])
@pytest.mark.parametrize("change", ["reverse", "increase"])
def test_changed_observations_recover_from_saturated_start(mode, change):
    y = torch.linspace(2, 2.1, 40, dtype=torch.float64)
    se = torch.ones_like(y)
    initial, target = (y, -y) if change == "reverse" else (y * 0.001, y)
    first = md_ash_posterior_means(None, initial, se, scale=[0, 0.3, 1, 3], sign_intercept=mode)
    saved = deepcopy(first.model_param)
    continued = md_ash_posterior_means(None, target, se, model_param=first.model_param)
    fresh = md_ash_posterior_means(None, target, se, scale=[0, 0.3, 1, 3], sign_intercept=mode)
    gap = continued.loss + continued.regularization_loss - fresh.loss - fresh.regularization_loss
    assert gap < 1e-4
    torch.testing.assert_close(continued.post_mean, fresh.post_mean, rtol=1e-4, atol=1e-5)
    assert continued.model_param["optimization"]["continuation_start"] != "saved"
    before = predict_md_ash(None, initial, se, model_param=saved)
    after = predict_md_ash(None, initial, se, model_param=first.model_param)
    torch.testing.assert_close(before.post_mean, after.post_mean, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"covariate_roles": "signed_evidence"}, "covariate_roles"),
        ({"covariate_roles": ["both"]}, "one valid"),
        ({"max_iter": 0}, "fresh fit"),
        ({"max_iter": 1.5}, "integer"),
        ({"penalty": 0.5}, "penalty"),
        ({"coefficient_prior_scale": 0}, "coefficient_prior_scale"),
        ({"scale": [0, 1, 0.5]}, "strictly increasing"),
    ],
)
def test_invalid_fit_contracts_fail_early(kwargs, message):
    X, y, se = data()
    with pytest.raises(ValueError, match=message):
        md_ash_posterior_means(X, y, se, **kwargs)


def test_missing_feature_requires_external_preparation():
    X, y, se = data()
    state = fitted().model_param
    X[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite prepared"):
        md_ash_posterior_means(X, y, se)
    with pytest.raises(ValueError, match="finite prepared"):
        predict_md_ash(X, model_param=state)


def test_saved_roles_grid_and_version_cannot_change_silently():
    X, y, se = data()
    state = fitted().model_param
    for kwargs, match in [({"covariate_roles": "both"}, "differs"), ({"scale": [0, 1, 2]}, "differs")]:
        with pytest.raises(ValueError, match=match):
            md_ash_posterior_means(X, y, se, model_param=state, max_iter=0, **kwargs)
    historical = deepcopy(state)
    historical["version"] = 4
    with pytest.raises(ValueError, match="Incompatible"):
        predict_md_ash(X, model_param=historical)


def test_saved_conditionals_must_have_finite_parameters():
    state = fitted().model_param
    state["state_dict"]["sign_intercepts"][0] = float("nan")
    with pytest.raises(FloatingPointError, match="coefficient"):
        predict_md_ash(data()[0], model_param=state)
