"""Analytic half-normal component likelihoods and stable truncated moments."""

import math

import torch
from torch import Tensor

from cebmf_torch.cebnm.lcash import _validate_grid
from cebmf_torch.utils.maths import _logcdf_normal, _logpdf_normal, _normal_tail_denominators

_LOG_SQRT_2PI = math.log(2 * math.pi) / 2


def positive_normal_moments(standardized_mean: Tensor, s: Tensor, *, return_variance=False):
    """Moments of N(s * standardized_mean, s**2) conditional on being positive.

    Inputs broadcast and require positive s. The continued fraction preserves
    small moments when the untruncated mean is far below zero. Direct variance
    calculation preserves uncertainty when the mean is far above zero.
    """
    t = -standardized_mean
    tail = t >= 4
    tail_t = torch.where(tail, t, torch.full_like(t, 4))
    d2, d3 = _normal_tail_denominators(tail_t)
    tail_mean = s / d2
    tail_mean2 = 2 * tail_mean * (s / d3)

    regular_t = torch.where(tail, torch.zeros_like(t), t)
    # erfcx avoids subtracting log densities for nonnegative t. On the
    # negative side the ordinary log-CDF expression has no cancellation.
    nonnegative_t = regular_t.clamp_min(0)
    positive_ratio = math.sqrt(2 / math.pi) / torch.special.erfcx(nonnegative_t / math.sqrt(2))
    negative_t = regular_t.clamp_max(0)
    negative_ratio = torch.exp(-0.5 * negative_t.square() - _LOG_SQRT_2PI - _logcdf_normal(-negative_t))
    ratio = torch.where(regular_t >= 0, positive_ratio, negative_ratio)
    unit_mean = ratio - regular_t
    regular_mean = s * unit_mean
    regular_mean2 = s * (s * (1 - regular_t * unit_mean))
    mean = torch.where(tail, tail_mean, regular_mean)
    second = torch.where(tail, tail_mean2, regular_mean2)
    if return_variance:
        variance = torch.where(tail, tail_mean2 - tail_mean.square(), s.square() * (1 - ratio * unit_mean)).clamp_min(
            0
        )
        return mean, second, variance
    return mean, second


def _inputs(x, s, scale):
    x = torch.as_tensor(x, dtype=torch.float64, device="cpu")
    s = torch.as_tensor(s, dtype=torch.float64, device="cpu")
    scale = torch.as_tensor(scale, dtype=torch.float64, device="cpu")
    if x.ndim != 1 or s.shape != x.shape or x.numel() == 0:
        raise ValueError("Effects and standard errors must be nonempty matching vectors.")
    if not torch.isfinite(x).all() or not torch.isfinite(s).all() or (s <= 0).any():
        raise ValueError("Effects must be finite with finite positive standard errors.")
    _validate_grid(scale)
    return x, s, scale


def get_data_loglik_halfnormal_torch(x, s, scale):
    """Return log densities in order [spike, negative_s1, positive_s1, ...].

    Positive scales are standard deviations before truncation at zero.
    Each signed component has unit mass on its half-line.
    """
    x, s, scale = _inputs(x, s, scale)
    spike = _logpdf_normal(x, torch.zeros_like(x), s)[:, None]
    marginal_sd = torch.hypot(s[:, None], scale[1:])
    z = (x / s)[:, None] * (scale[1:] / marginal_sd)
    normal = _logpdf_normal(x[:, None], x.new_zeros(()), marginal_sd)
    signed = normal[:, :, None] + math.log(2) + _logcdf_normal(torch.stack((-z, z), dim=2))
    result = torch.cat((spike, signed.flatten(1)), dim=1)
    if not torch.isfinite(result).all():
        raise FloatingPointError(
            "Half-normal log densities are non-finite. Check observation and standard error scales."
        )
    return result


def halfnormal_moments(x, s, scale, *, return_variance=False):
    """Return signed component moments without the spike column."""
    x, s, scale = _inputs(x, s, scale)
    marginal_sd = torch.hypot(s[:, None], scale[1:])
    ratio = scale[1:] / marginal_sd
    z = (x / s)[:, None] * ratio
    post_sd = (s[:, None] * ratio)[:, :, None]
    signs = x.new_tensor([-1, 1])
    mean, second, variance = positive_normal_moments(z[:, :, None] * signs, post_sd, return_variance=True)
    values = ((mean * signs).flatten(1), second.flatten(1), variance.flatten(1))
    return values if return_variance else values[:2]


def truncated_normal_log_predictive(target, target_se, mean, sd, sign):
    """Convolve a signed truncated normal with target observation error.

    ``mean`` and ``sd`` describe the normal before truncation at zero.
    Inputs broadcast. The erfcx branch combines cancelling quadratic terms
    analytically when both normalizing CDF arguments are in the negative tail.
    """
    total = torch.hypot(sd, target_se)
    a = sign * mean / sd
    u = (target_se / total) * a + sign * (sd / total) * (target / target_se)
    result = _logpdf_normal(target, mean, total) + _logcdf_normal(u) - _logcdf_normal(a)
    tail = (a < 0) & (u < 0)
    # Clamp only unused erfcx arguments; erfcx overflows for large negatives.
    log_ratio = (
        torch.special.erfcx(-u.clamp_max(0) / math.sqrt(2)).log()
        - torch.special.erfcx(-a.clamp_max(0) / math.sqrt(2)).log()
    )
    stable = _logpdf_normal(target, torch.zeros_like(target), target_se) + target_se.log() - total.log() + log_ratio
    return torch.where(tail, stable, result)
