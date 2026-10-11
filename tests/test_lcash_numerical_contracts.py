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
def test_transient_nonfinite_gradient_does_not_return_a_fit(model_class, solver, value, monkeypatch):
    monkeypatch.setattr(
        lcash,
        "ash",
        lambda *args, **kwargs: SimpleNamespace(pi=torch.tensor([0.6, 0.3, 0.1]), scale=torch.tensor([0.0, 1.0, 2.0])),
    )
    original_log_probs = model_class.log_probs
    calls = []
    handles = []

    def poison_first_gradient(gradient):
        calls.append(True)
        return torch.full_like(gradient, value) if len(calls) == 1 else gradient

    def attach_hook(model, features):
        result = original_log_probs(model, features)
        if not handles:
            assert torch.isfinite(result).all()
            handles.append(next(model.parameters()).register_hook(poison_first_gradient))
        return result

    monkeypatch.setattr(model_class, "log_probs", attach_hook)
    x = torch.linspace(-1.0, 1.0, 5)
    try:
        with pytest.raises(FloatingPointError, match="Conditional parameter.*finite"):
            solver(x[:, None], x, torch.ones_like(x), n_epochs=3, batch_size=len(x), verbose=False)
    finally:
        for handle in handles:
            handle.remove()
    assert len(calls) == 3


@pytest.mark.parametrize(
    "solver,value,error",
    [
        (lcash_posterior_means, math.nan, "Conditional parameter.*finite"),
        (lcash_posterior_means, math.inf, "Conditional parameter.*finite"),
        (po_lcash_posterior_means, math.nan, "Conditional parameter.*finite"),
        (po_lcash_posterior_means, math.inf, "Conditional parameter.*finite"),
        (po_lcash_posterior_means, 89.0, "log probabilities.*finite"),
        (po_lcash_posterior_means, math.log(torch.finfo(torch.float32).max * 0.75), "log probabilities.*finite"),
    ],
)
def test_invalid_final_adam_step_is_not_returned_as_a_fit(solver, value, error, monkeypatch):
    monkeypatch.setattr(
        lcash,
        "ash",
        lambda *args, **kwargs: SimpleNamespace(
            pi=torch.tensor([0.6, 0.2, 0.15, 0.05]), scale=torch.tensor([0.0, 1.0, 2.0, 4.0])
        ),
    )
    original_step = torch.optim.Adam.step
    calls = []

    def invalid_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        with torch.no_grad():
            # The last parameter is the LC bias or the PO log gap.
            optimizer.param_groups[1]["params"][-1].fill_(value)
        calls.append(True)
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", invalid_step)
    x = torch.linspace(-2.0, 2.0, 8)
    with pytest.raises(FloatingPointError, match=error):
        solver(x[:, None], x, torch.ones_like(x), n_epochs=1, batch_size=len(x), verbose=False)
    assert calls == [True]
