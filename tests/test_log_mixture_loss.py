"""Contracts for the log-space loss and CASH's probability-input wrapper."""

import pytest
import torch

from cebmf_torch.cebnm.cash_solver import pen_loglik_loss_from_log_pi, pen_loglik_loss


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


def test_zero_null_coefficient_accepts_negative_infinite_null_log_weight():
    log_pi = torch.tensor([[-torch.inf, 0.0], [-torch.inf, 0.0]], dtype=torch.float64, requires_grad=True)
    log_lik = torch.tensor([[4.0, -2.0], [10.0, 3.0]], dtype=torch.float64)
    loss = pen_loglik_loss_from_log_pi(log_pi, log_lik)
    gradient = torch.autograd.grad(loss, log_pi)[0]
    torch.testing.assert_close(loss, -log_lik[:, 1].sum(), rtol=0, atol=0)
    torch.testing.assert_close(gradient, torch.tensor([[0.0, -1.0], [0.0, -1.0]], dtype=gradient.dtype))
