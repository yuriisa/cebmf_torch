"""Regression contracts for integer controls and conditional batch penalties."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cebmf_torch.cebnm.lcash import LcashNet, PropOddsLcashNet, lcash_posterior_means, po_lcash_posterior_means

lcash = importlib.import_module("cebmf_torch.cebnm.lcash")


@pytest.mark.parametrize("solver", [lcash_posterior_means, po_lcash_posterior_means])
@pytest.mark.parametrize("integer_type", [np.int32, np.int64])
@pytest.mark.parametrize("epochs", [0, 2])
def test_numpy_integer_controls_match_python_integers(solver, integer_type, epochs, monkeypatch):
    monkeypatch.setattr(
        lcash,
        "ash",
        lambda *args, **kwargs: SimpleNamespace(pi=torch.tensor([0.7, 0.3]), scale=torch.tensor([0.0, 1.0])),
    )
    x = torch.linspace(-2, 2, 7)
    inputs = (x[:, None], x, torch.ones_like(x))
    expected = solver(*inputs, n_epochs=epochs, batch_size=3, verbose=False)
    actual = solver(*inputs, n_epochs=integer_type(epochs), batch_size=integer_type(3), verbose=False)
    for field in ("post_mean", "post_mean2", "post_sd", "pi_np"):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)
    assert actual.loss == expected.loss


@pytest.mark.parametrize("model_class", [LcashNet, PropOddsLcashNet])
@pytest.mark.parametrize("penalty", [1.0, 1.5, 10.0])
def test_training_uses_full_sample_penalty_for_incomplete_batch(model_class, penalty, monkeypatch):
    calls = []
    original_loss = lcash.pen_loglik_loss_from_log_pi

    def record_loss(log_pi, marginal_log_lik, null_coefficient=0.0):
        actual = original_loss(log_pi, marginal_log_lik, null_coefficient)
        reference_log_pi = log_pi.detach().clone().requires_grad_()
        expected = -torch.logsumexp(reference_log_pi + marginal_log_lik, dim=1).sum()
        expected -= ((penalty - 1) / 7) * reference_log_pi[:, 0].sum()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, log_pi, retain_graph=True)[0],
            torch.autograd.grad(expected, reference_log_pi)[0],
        )
        calls.append((len(log_pi), null_coefficient))
        return actual

    monkeypatch.setattr(lcash, "pen_loglik_loss_from_log_pi", record_loss)
    model = model_class(1, 3, log_pi_init=torch.tensor([0.6, 0.3, 0.1]).log())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.linspace(-2, 2, 7)
    lcash._train_model(
        model,
        optimizer,
        x[:, None],
        x,
        torch.ones_like(x),
        torch.tensor([0.0, 1.0, 2.0]),
        n_epochs=2,
        batch_size=3,
        penalty=penalty,
        verbose=False,
        label="test",
    )
    coefficient = (penalty - 1) / len(x)
    assert calls == [(3, coefficient), (3, coefficient), (1, coefficient)] * 2
