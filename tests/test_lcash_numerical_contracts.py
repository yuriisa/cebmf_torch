"""Numerical and public-call contracts for the conditional solvers."""

import importlib
import math
from decimal import Decimal, localcontext
from types import SimpleNamespace

import pytest
import torch

from cebmf_torch.cebnm.lcash import LcashNet, PropOddsLcashNet, lcash_posterior_means, po_lcash_posterior_means

lcash = importlib.import_module("cebmf_torch.cebnm.lcash")
MODELS = [LcashNet, PropOddsLcashNet]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gap_switch_matches_decimal_values_and_derivatives(dtype):
    eps = torch.finfo(dtype).eps
    switch = torch.tensor(math.log(eps), dtype=dtype)
    neighbors = [torch.nextafter(switch, torch.tensor(target, dtype=dtype)).item() for target in (-math.inf, math.inf)]
    values = [-90.0, -20.0, neighbors[0], switch.item(), neighbors[1], -1.0, 0.0, 2.0, 5.0]
    h = torch.tensor(values, dtype=dtype, requires_grad=True)
    actual = lcash._log_gap_mass(h)
    derivative = torch.autograd.grad(actual.sum(), h)[0]
    references = []
    with localcontext() as context:
        context.prec = 120
        one = Decimal(1)
        for value in h.detach().tolist():
            gap = Decimal.from_float(value).exp()
            references.append((float((one - (-gap).exp()).ln()), float(gap / (gap.exp() - one))))
    reference = torch.tensor(references, dtype=dtype)
    torch.testing.assert_close(actual, reference[:, 0], atol=8 * eps, rtol=8 * eps)
    torch.testing.assert_close(derivative, reference[:, 1], atol=8 * eps, rtol=8 * eps)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_po_rejects_cumulative_cutpoint_overflow_with_finite_gaps(dtype):
    model = PropOddsLcashNet(1, 4).to(dtype=dtype)
    with torch.no_grad():
        model.w.zero_()
        model.delta_1.zero_()
        model.delta_gaps.fill_(math.log(torch.finfo(dtype).max * 0.75))
    assert torch.isfinite(model.delta_gaps.exp()).all()
    x = torch.zeros(2, dtype=dtype)
    with pytest.raises(FloatingPointError, match="log probabilities.*finite"):
        lcash._compute_posteriors(
            model, x[:, None], x, torch.ones_like(x), torch.tensor([0.0, 1.0, 2.0, 4.0], dtype=dtype)
        )


@pytest.mark.parametrize(
    "model_class,solver", zip(MODELS, [lcash_posterior_means, po_lcash_posterior_means], strict=True)
)
@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_nonfinite_trial_gradient_restores_model_and_does_not_return_a_fit(model_class, solver, value, monkeypatch):
    monkeypatch.setattr(
        lcash,
        "ash",
        lambda *args, **kwargs: SimpleNamespace(pi=torch.tensor([0.6, 0.3, 0.1]), scale=torch.tensor([0.0, 1.0, 2.0])),
    )
    original_log_probs = model_class.log_probs
    calls = []
    handles = []
    initial_parameters = []
    models = []

    def poison_first_gradient(gradient):
        calls.append(True)
        return torch.full_like(gradient, value) if len(calls) == 2 else gradient

    def attach_hook(model, features):
        result = original_log_probs(model, features)
        if not handles:
            assert torch.isfinite(result).all()
            models.append(model)
            initial_parameters.extend(parameter.detach().clone() for parameter in model.parameters())
            handles.append(next(model.parameters()).register_hook(poison_first_gradient))
        return result

    monkeypatch.setattr(model_class, "log_probs", attach_hook)
    x = torch.linspace(-1.0, 1.0, 5)
    try:
        with pytest.raises(FloatingPointError, match="gradient.*finite"):
            solver(x[:, None], x, torch.ones_like(x), max_iter=3, verbose=False)
    finally:
        for handle in handles:
            handle.remove()
    assert len(calls) == 2
    for parameter, initial in zip(models[0].parameters(), initial_parameters, strict=True):
        torch.testing.assert_close(parameter, initial, rtol=0, atol=0)


@pytest.mark.parametrize(
    "solver,value,error",
    [
        (lcash_posterior_means, math.nan, "Conditional .*finite"),
        (lcash_posterior_means, math.inf, "Conditional .*finite"),
        (po_lcash_posterior_means, math.nan, "Conditional .*finite"),
        (po_lcash_posterior_means, math.inf, "Conditional .*finite"),
        (po_lcash_posterior_means, 710.0, "Conditional .*finite"),
        (po_lcash_posterior_means, math.log(torch.finfo(torch.float64).max * 0.75), "Conditional .*finite"),
    ],
)
def test_invalid_final_lbfgs_step_is_not_returned_as_a_fit(solver, value, error, monkeypatch):
    monkeypatch.setattr(
        lcash,
        "ash",
        lambda *args, **kwargs: SimpleNamespace(
            pi=torch.tensor([0.6, 0.2, 0.15, 0.05]), scale=torch.tensor([0.0, 1.0, 2.0, 4.0])
        ),
    )
    original_step = torch.optim.LBFGS.step
    calls = []

    def invalid_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        with torch.no_grad():
            # The last parameter is the LC bias or the PO log gap.
            optimizer.param_groups[0]["params"][-1].fill_(value)
        calls.append(True)
        return result

    monkeypatch.setattr(torch.optim.LBFGS, "step", invalid_step)
    x = torch.linspace(-2.0, 2.0, 8)
    with pytest.raises(FloatingPointError, match=error):
        solver(x[:, None], x, torch.ones_like(x), max_iter=1, verbose=False)
    assert calls == [True]


@pytest.mark.parametrize("solver", [lcash_posterior_means, po_lcash_posterior_means])
def test_output_cast_overflow_is_not_returned_as_a_fit(solver, monkeypatch):
    monkeypatch.setattr(
        lcash,
        "ash",
        lambda *args, **kwargs: SimpleNamespace(
            pi=torch.tensor([0.5, 0.5], dtype=torch.float64),
            scale=torch.tensor([0.0, 1e20], dtype=torch.float64),
        ),
    )
    observed = torch.full((2,), 1e20, dtype=torch.float64)
    with pytest.raises(FloatingPointError, match="Returned conditional post_mean2"):
        solver(torch.zeros(2, 1), observed, torch.ones_like(observed), max_iter=0, verbose=False)


@pytest.mark.parametrize("solver", [lcash_posterior_means, po_lcash_posterior_means])
def test_output_cast_must_preserve_distinct_component_scales(solver, monkeypatch):
    monkeypatch.setattr(
        lcash,
        "ash",
        lambda *args, **kwargs: SimpleNamespace(pi=torch.tensor([0.5, 0.5]), scale=torch.tensor([0.0, 1.0])),
    )
    features = torch.zeros(2, 1)
    state = solver(features, torch.zeros(2), torch.ones(2), max_iter=0, verbose=False).model_param
    state["scale"] = torch.tensor([0.0, 1e-50], dtype=torch.float64)
    observed = torch.full((2,), 1e-49, dtype=torch.float64)
    standard_error = torch.full((2,), 1e-50, dtype=torch.float64)
    with pytest.raises(ValueError, match="strictly increasing"):
        solver(features, observed, standard_error, model_param=state, max_iter=0, verbose=False)
