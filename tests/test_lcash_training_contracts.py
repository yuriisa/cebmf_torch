"""L-BFGS controls, deterministic fitting, and informative-covariate recovery."""

import importlib
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cebmf_torch.cebnm.lcash import LcashNet, PropOddsLcashNet, lcash_posterior_means, po_lcash_posterior_means

lcash = importlib.import_module("cebmf_torch.cebnm.lcash")
SOLVERS = [lcash_posterior_means, po_lcash_posterior_means]


@pytest.fixture
def fixed_grid(monkeypatch):
    monkeypatch.setattr(
        lcash,
        "ash",
        lambda *args, **kwargs: SimpleNamespace(pi=torch.tensor([0.6, 0.4]), scale=torch.tensor([0.0, 3.0])),
    )


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("integer_type", [np.int32, np.int64])
@pytest.mark.parametrize("iterations", [0, 2])
def test_numpy_integer_controls_match_python_integers(solver, integer_type, iterations, fixed_grid):
    x = torch.linspace(-2, 2, 7)
    inputs = (x[:, None], x, torch.ones_like(x))
    expected = solver(*inputs, max_iter=iterations, verbose=False)
    actual = solver(*inputs, max_iter=integer_type(iterations), verbose=False)
    for field in ("post_mean", "post_mean2", "post_sd", "pi_np"):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)
    assert actual.loss == expected.loss


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("invalid", [-1, 1.5, float("nan"), float("inf")])
def test_invalid_iteration_budget_is_rejected(solver, invalid, fixed_grid):
    x = torch.linspace(-2, 2, 7)
    with pytest.raises((TypeError, ValueError), match="max_iter"):
        solver(x[:, None], x, torch.ones_like(x), max_iter=invalid, verbose=False)


@pytest.mark.parametrize("model_class", [LcashNet, PropOddsLcashNet])
@pytest.mark.parametrize("initial_weights", [None, [0.6, 0.3, 0.1]])
def test_model_construction_preserves_global_rng(model_class, initial_weights):
    log_pi = None if initial_weights is None else torch.tensor(initial_weights).log()
    before = torch.random.get_rng_state().clone()
    model = model_class(2, 3, log_pi_init=log_pi)
    assert torch.equal(torch.random.get_rng_state(), before)
    slopes = model.linear.weight if isinstance(model, LcashNet) else model.w
    assert torch.count_nonzero(slopes) == 0


@pytest.mark.parametrize("solver", SOLVERS)
def test_fitting_preserves_global_rng(solver, fixed_grid):
    x = torch.linspace(-2, 2, 32)
    before = torch.random.get_rng_state().clone()
    solver(x[:, None], x, torch.ones_like(x), max_iter=2, verbose=False)
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize("solver", SOLVERS)
def test_informative_covariate_recovery_with_few_observations(solver, fixed_grid):
    generator = torch.Generator().manual_seed(410)
    x = torch.randn(500, generator=generator)
    x = (x - x.mean()) / x.std(correction=0)
    slab = torch.rand(500, generator=generator) < torch.sigmoid(1.5 * x - 0.5)
    truth = 3 * slab * torch.randn(500, generator=generator)
    observed = truth + torch.randn(500, generator=generator)
    first = solver(x[:, None], observed, torch.ones_like(x), max_iter=0, verbose=False)
    fitted = solver(x[:, None], observed, torch.ones_like(x), model_param=first.model_param, verbose=False)
    state = fitted.model_param["state_dict"]
    if solver is lcash_posterior_means:
        slope = (state["linear.weight"][1] - state["linear.weight"][0]).item()
    else:
        slope = state["w"].item()
    # This range admits sampling error but rejects a fit that barely leaves zero.
    assert 0.8 < slope < 2.2
    assert fitted.loss < first.loss - 5
    assert torch.mean((fitted.post_mean - truth).square()) < torch.mean((first.post_mean - truth).square())


@pytest.mark.parametrize("model_class", [LcashNet, PropOddsLcashNet])
@pytest.mark.parametrize("penalty", [1.0, 1.5, 10.0])
def test_full_data_objective_has_only_the_null_penalty(model_class, penalty, monkeypatch):
    calls = []
    original_step = torch.optim.LBFGS.step
    model = model_class(1, 3, log_pi_init=torch.tensor([0.6, 0.3, 0.1]).log()).double()
    with torch.no_grad():
        slopes = model.linear.weight if isinstance(model, LcashNet) else model.w
        slopes.fill_(0.5)
        if isinstance(model, LcashNet):
            slopes[0].zero_()
    x = torch.linspace(-2, 2, 7, dtype=torch.float64)
    scale = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    log_lik = lcash.get_data_loglik_normal_torch(x, torch.ones_like(x), torch.zeros_like(scale), scale)

    def inspect_first_closure(optimizer, closure):
        actual = closure()
        actual_gradients = [parameter.grad.detach().clone() for parameter in model.parameters()]
        log_pi = model.log_probs(x[:, None])
        expected = -torch.logsumexp(log_pi + log_lik, dim=1).sum()
        expected -= ((penalty - 1) / len(x)) * log_pi[:, 0].sum()
        expected /= len(x)
        gradients = torch.autograd.grad(expected, tuple(model.parameters()))
        torch.testing.assert_close(actual, expected)
        for actual_gradient, gradient in zip(actual_gradients, gradients, strict=True):
            torch.testing.assert_close(actual_gradient, gradient)
        calls.append(True)
        return original_step(optimizer, closure)

    monkeypatch.setattr(torch.optim.LBFGS, "step", inspect_first_closure)
    lcash._train_model(
        model,
        x[:, None],
        x,
        torch.ones_like(x),
        scale,
        max_iter=2,
        penalty=penalty,
        verbose=False,
        label="test",
    )
    assert calls == [True]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gap_mass_asymptotes_have_finite_gradients(dtype):
    high = 80 if dtype == torch.float32 else 700
    log_gap = torch.tensor([-1000.0, high], dtype=dtype, requires_grad=True)
    result = lcash._log_gap_mass(log_gap)
    gradient = torch.autograd.grad(result.sum(), log_gap)[0]
    torch.testing.assert_close(result, torch.tensor([-1000.0, 0.0], dtype=dtype), rtol=0, atol=0)
    torch.testing.assert_close(gradient, torch.tensor([1.0, 0.0], dtype=dtype), rtol=0, atol=0)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_training_state_keeps_float64_and_outputs_keep_requested_device(solver, device, fixed_grid):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    x = torch.linspace(-2, 2, 16, device=device)
    result = solver(x[:, None], x, torch.ones_like(x), max_iter=2, device=device, verbose=False)
    for key in ("scale", "feature_mean", "feature_sd"):
        assert result.model_param[key].dtype == torch.float64
    for value in result.model_param["state_dict"].values():
        assert value.dtype == torch.float64
        assert value.device.type == ("cpu" if device == "mps" else device)
    for field in ("post_mean", "post_mean2", "post_sd", "pi_np", "scale"):
        value = getattr(result, field)
        assert value.dtype == torch.float32
        assert value.device.type == device


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("options", [{}, {"max_iter": None}, {"max_iter": 0}, {"max_iter": 7}])
def test_standalone_iteration_default_and_zero_prediction(solver, options, fixed_grid, monkeypatch):
    calls = []

    def record_training(model, X, betahat, sebetahat, scale, max_iter, penalty, verbose, label):
        calls.append(max_iter)

    monkeypatch.setattr(lcash, "_train_model", record_training)
    x = torch.linspace(-2, 2, 7)
    solver(x[:, None], x, torch.ones_like(x), verbose=False, **options)
    requested = options.get("max_iter")
    assert calls == ([] if requested == 0 else [500 if requested is None else requested])


@pytest.mark.parametrize("model_class", [LcashNet, PropOddsLcashNet])
@pytest.mark.parametrize("stop", ["budget", "small_step"])
def test_incomplete_fit_warning_identifies_stop_reason(model_class, stop, monkeypatch):
    if stop == "small_step":
        original_init = torch.optim.LBFGS.__init__

        def stop_before_update(optimizer, *args, **kwargs):
            kwargs["tolerance_change"] = 1.0
            original_init(optimizer, *args, **kwargs)

        monkeypatch.setattr(torch.optim.LBFGS, "__init__", stop_before_update)
    model = model_class(1, 2, log_pi_init=torch.tensor([0.6, 0.4]).log()).double()
    x = torch.linspace(-3, 3, 100, dtype=torch.float64)
    scale = torch.tensor([0.0, 3.0], dtype=torch.float64)
    reason = "iteration or evaluation budget" if stop == "budget" else "small step or objective change"
    with pytest.warns(RuntimeWarning, match=reason):
        lcash._train_model(
            model,
            x[:, None],
            x,
            torch.ones_like(x),
            scale,
            max_iter=1 if stop == "budget" else 500,
            penalty=10,
            verbose=False,
            label="test",
        )
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_converged_fit_has_no_incomplete_fit_warning():
    model = LcashNet(1, 2, log_pi_init=torch.tensor([0.6, 0.4]).log()).double()
    x = torch.linspace(-3, 3, 100, dtype=torch.float64)
    scale = torch.tensor([0.0, 3.0], dtype=torch.float64)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        lcash._train_model(
            model,
            x[:, None],
            x,
            torch.ones_like(x),
            scale,
            max_iter=500,
            penalty=10,
            verbose=False,
            label="test",
        )
    assert not captured
    assert max(parameter.grad.abs().max().item() for parameter in model.parameters()) <= 1e-7
