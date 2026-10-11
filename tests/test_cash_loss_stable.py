"""The shared mixture loss remains finite for large log densities."""

import torch

from cebmf_torch.cebnm.cash_solver import pen_loglik_loss_from_log_pi


def test_loss_finite_for_large_log_likelihoods():
    # Small SEs push convolved log-densities well past the float32 exp() limit.
    torch.manual_seed(0)
    B, K = 16, 5
    pred_pi = torch.softmax(torch.randn(B, K), dim=1)
    marginal_log_lik = torch.full((B, K), 90.0)  # exp(90) overflows float32
    loss = pen_loglik_loss_from_log_pi(pred_pi.log(), marginal_log_lik, null_coefficient=0.5)
    assert torch.isfinite(loss), loss


def test_loss_matches_naive_in_safe_regime():
    # In this regime, direct summation is an independent reference.
    torch.manual_seed(1)
    B, K = 8, 4
    pred_pi = torch.softmax(torch.randn(B, K), dim=1)
    mll = torch.randn(B, K) * 0.5  # small -> no overflow
    eps = 1e-10
    naive_first = torch.log(torch.clamp((pred_pi * torch.exp(mll)).sum(1), min=eps)).sum()
    naive = -(naive_first + (1.5 - 1) * torch.log(torch.clamp(pred_pi[:, 0], min=eps)).sum())
    got = pen_loglik_loss_from_log_pi(pred_pi.log(), mll, null_coefficient=0.5)
    assert abs(float(got) - float(naive)) < 1e-4, (float(got), float(naive))


def test_loss_is_differentiable():
    B, K = 6, 3
    logits = torch.randn(B, K, requires_grad=True)
    pred_pi = torch.softmax(logits, dim=1)
    mll = torch.full((B, K), 50.0)
    loss = pen_loglik_loss_from_log_pi(pred_pi.log(), mll, null_coefficient=1.0)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
