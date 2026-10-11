"""Contracts for the log-space loss and CASH's probability-input wrapper."""

import pytest
import torch

from cebmf_torch.cebnm.cash_solver import pen_loglik_loss, pen_loglik_loss_from_log_pi


def _cash_loss_reference(pred_pi, marginal_log_lik, penalty, epsilon):
    """Evaluate CASH's loss formula with a separate calculation of the null term."""
    log_pi = torch.log(torch.clamp(pred_pi, min=epsilon))
    value = torch.logsumexp(log_pi + marginal_log_lik, dim=1).sum()
    if penalty > 1:
        log_pi0 = torch.log(torch.clamp(pred_pi[:, 0], min=epsilon))
        value = value + (penalty - 1) * log_pi0.sum()
    return -value


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("penalty", [-2.0, 0.0, 1.0, 1.5, 10.0])
@pytest.mark.parametrize("epsilon", [1e-10, 1e-6])
def test_cash_wrapper_matches_reference_values_and_gradients(dtype, penalty, epsilon):
    pi = torch.tensor(
        [[0.0, 1e-12, 0.4, 0.6], [0.2, 0.3, 0.4, 0.1], [epsilon, epsilon, 0.5, 0.5], [1e-4, 0.2, 0.3, 0.4999]],
        dtype=dtype,
        requires_grad=True,
    )
    log_lik = torch.tensor(
        [[90.0, 80.0, -200.0, 0.0], [-1.0, -2.0, -3.0, 4.0], [-100.0, 0.0, 5.0, 10.0], [1.0, -50.0, 5.0, -20.0]],
        dtype=dtype,
        requires_grad=True,
    )
    reference = _cash_loss_reference(pi, log_lik, penalty, epsilon)
    actual = pen_loglik_loss(pi, log_lik, penalty=penalty, epsilon=epsilon)
    reference_grad = torch.autograd.grad(reference, (pi, log_lik))
    actual_grad = torch.autograd.grad(actual, (pi, log_lik))
    eps = torch.finfo(dtype).eps
    # The shared null column is strided; its sum can differ by one rounding unit.
    torch.testing.assert_close(actual, reference, rtol=8 * eps, atol=8 * eps)
    for got, expected in zip(actual_grad, reference_grad, strict=True):
        torch.testing.assert_close(got, expected, rtol=8 * eps, atol=8 * eps)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_log_weights_preserve_small_components_and_gradients(dtype):
    log_pi = torch.tensor([[-1000.0, 0.0]], dtype=dtype, requires_grad=True)
    log_lik = torch.tensor([[1000.0, 0.0]], dtype=dtype)
    loss = pen_loglik_loss_from_log_pi(log_pi, log_lik, null_coefficient=0.0)
    gradient = torch.autograd.grad(loss, log_pi)[0]
    torch.testing.assert_close(loss, -torch.tensor(2.0, dtype=dtype).log())
    torch.testing.assert_close(gradient, torch.tensor([[-0.5, -0.5]], dtype=dtype))


def test_zero_null_coefficient_accepts_negative_infinite_null_log_weight():
    log_pi = torch.tensor([[-torch.inf, 0.0], [-torch.inf, 0.0]], dtype=torch.float64, requires_grad=True)
    log_lik = torch.tensor([[4.0, -2.0], [10.0, 3.0]], dtype=torch.float64)
    loss = pen_loglik_loss_from_log_pi(log_pi, log_lik)
    gradient = torch.autograd.grad(loss, log_pi)[0]
    torch.testing.assert_close(loss, -log_lik[:, 1].sum(), rtol=0, atol=0)
    torch.testing.assert_close(gradient, torch.tensor([[0.0, -1.0], [0.0, -1.0]], dtype=gradient.dtype))


@pytest.mark.parametrize("n", [2, 7, 13])
@pytest.mark.parametrize("penalty", [1.0, 1.5, 10.0])
def test_null_coefficient_uses_full_count_across_incomplete_batches(n, penalty):
    logits = torch.linspace(-2.0, 2.0, n * 3, dtype=torch.float64).reshape(n, 3).requires_grad_()
    log_lik = torch.linspace(1.0, -3.0, n * 3, dtype=logits.dtype).reshape(n, 3)
    log_pi = logits.log_softmax(dim=1)
    coefficient = (penalty - 1) / n
    full = pen_loglik_loss_from_log_pi(log_pi, log_lik, coefficient)
    partitioned = sum(
        pen_loglik_loss_from_log_pi(log_pi[start : start + 3], log_lik[start : start + 3], coefficient)
        for start in range(0, n, 3)
    )
    full_gradient = torch.autograd.grad(full, logits, retain_graph=True)[0]
    batch_gradient = torch.autograd.grad(partitioned, logits)[0]
    torch.testing.assert_close(partitioned, full, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(batch_gradient, full_gradient, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("penalty", [1.0, 1.5, 10.0])
def test_constant_weights_match_ash_penalty(penalty):
    pi = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64, requires_grad=True)
    log_lik = torch.linspace(-2.0, 2.0, 21, dtype=pi.dtype).reshape(7, 3)
    n = len(log_lik)
    actual = pen_loglik_loss_from_log_pi(pi.log().expand(n, -1), log_lik, (penalty - 1) / n)
    expected = -(pi * log_lik.exp()).sum(dim=1).log().sum() - (penalty - 1) * pi[0].log()
    actual_gradient = torch.autograd.grad(actual, pi)[0]
    expected_gradient = torch.autograd.grad(expected, pi)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("n", [2, 7, 13])
@pytest.mark.parametrize("cash_penalty", [1.0, 1.5])
def test_conditional_nullweight_matches_cash_loss_and_gradient(n, cash_penalty):
    logits = torch.linspace(-1.0, 1.0, n * 3, dtype=torch.float64).reshape(n, 3).requires_grad_()
    log_lik = torch.linspace(2.0, -2.0, n * 3, dtype=logits.dtype).reshape(n, 3)
    pi = logits.softmax(dim=1)
    conditional_penalty = 1 + n * (cash_penalty - 1)
    expected = pen_loglik_loss(pi, log_lik, penalty=cash_penalty)
    actual = pen_loglik_loss_from_log_pi(logits.log_softmax(dim=1), log_lik, (conditional_penalty - 1) / n)
    expected_gradient = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual, logits)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-14, atol=1e-14)
