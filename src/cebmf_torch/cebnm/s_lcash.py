"""S-LC-ASH: Scaled Linear Covariate-mediated Adaptive Shrinkage.

Per-level spike-and-slab where the slab scale ``c_t`` is per-level and
everything else is shared:

.. math::

    \\beta_{g,t} \\sim p \\cdot \\delta_0 + (1 - p) \\sum_{k=1}^{K} w_k\\,
    \\mathcal{N}\\!\\left(0,\\ c_t^2\\sigma_k^2\\right)

* ``p`` (scalar) — shared spike weight (fraction of null genes, panel-wide).
* ``w_k``, ``sigma_k`` — shared slab weights and widths.
* ``c_t > 0`` — per-level scale on the slab widths. Hyperprior
  ``log c_t ~ N(mu_c, tau_c^2)`` with ``mu_c`` and ``tau_c^2`` empirical-Bayes
  fitted (jointly with everything else by Adam, by default).

The model is a one-dial per-level extension of pooled `ash` and matches the
sister-project ``beta_pool_scale`` parameterisation. The "LC" in the name
keeps the door open for covariate dependence on ``c_t`` in a future PR
(e.g. phenotype-class regulating ``log c`` linearly); the present
implementation has a single categorical covariate (level id; e.g. trait id, cohort id, tissue id).

Public API:

* :func:`s_lcash_posterior_means` — panel fit.
* :func:`s_lcash_new_level_posterior_means` — cold-start a new level given a panel-trained model.
* :class:`SLcashNet` — per-level + shared-parameter container (``nn.Module``).
* :func:`s_lcash_log_marginal` — per-observation marginal log-density kernel.
* :func:`s_lcash_compute_posteriors` — per-observation posterior moments kernel.

Notes on the level-2 hyperparameter ``tau_c^2``
-----------------------------------------------

The level-2 prior on ``log c_t`` is a free-mean Normal
``N(mu_c, tau_c^2)`` with ``mu_c`` and ``log tau_c`` learnable
parameters optimised jointly with everything else under a single
Adam loop. The minimised loss (negative log-prior, summed over the
``T`` levels) decomposes into two terms whose gradients on
``log_tau_c`` act in opposite directions:

* The normaliser ``+T * log(tau_c)``. Gradient ``+T``. In a
  minimisation Adam moves ``log_tau_c`` *down*; this is the term
  that drives ``tau_c -> 0``.
* The quadratic residual ``+sum_t (log c_t - mu_c)^2 / (2 tau_c^2)``.
  Gradient ``-sum_t (log c_t - mu_c)^2 / tau_c^2``. Adam moves
  ``log_tau_c`` *up*. This is the term that resists collapse, and
  it does so **only when per-level ``log c_t`` values diverge from
  ``mu_c``**.

The two terms balance at ``tau_c^2 = (1/T) sum_t (log c_t - mu_c)^2``
— the empirical-variance equilibrium. On panels where the data
prefers a homogeneous solution (every ``c_t`` close to the panel
mean), the quadratic term has nothing to push against, the
normaliser dominates, and ``tau_c`` is driven toward zero. The
``tau2_min`` parameter installs a soft floor (see below) to keep
the optimisation numerically well-behaved when this happens. The
practical advantage of joint Adam over closed-form alternating
empirical Bayes is dynamic stability: alternating EB sets
``tau_c^2`` to the empirical variance instantly each E-step and can
oscillate or overshoot toward the floor when per-level values are
momentarily clustered; joint Adam smooths the trajectory.

The floor is implemented smoothly as
``tau_c = sqrt(tau2_min + exp(2 * log_tau_c))`` rather than as a
hard ``clamp(min=sqrt(tau2_min))``. With the smooth floor,
gradients on ``log_tau_c`` flow at all values, so if per-level
heterogeneity later emerges the optimiser can recover ``tau_c``
above the floor; with a hard clamp the gradient through ``tau_c``
is zero below the floor and recovery is impossible.

This was observed by the CAESER 246-level validation in both
training directions: the per-level ``c_t`` values clustered tightly
around the panel mean (range ``[0.949, 0.966]`` on UKB-trained,
``[0.984, 0.986]`` on AGD-trained), and ``tau_c^2`` collapsed to
``~10^-6`` over a few hundred epochs. The marginal log-likelihood
decreased by tens of nats panel-total over a 1500-epoch fit,
matching the diagnostic signature of misspecified-prior collapse,
**but the predictive impact is negligible** (paired test LPD changes
by ``~10^-5``). The model degrades gracefully to "pooled ASH plus
shared spike" in this regime, which itself beats per-level ASH on
that panel.

The deployment workflow side-steps the issue by panel construction:
restricting the panel to well-pinned high-power levels (and adding
underpowered levels via :func:`s_lcash_new_level_posterior_means`) keeps the per-level
``c_t`` values diverse enough that ``tau_c^2`` stays well above the
floor. CAESER measured ``tau_c^2 = 0.115`` on the well-pinned 50-level
subset versus ``4 x 10^-4`` on the full 246-level fit.

A consequence of the **free-mean** parameterisation: any panel-wide
multiplicative shift in slab width is absorbed into ``mu_c`` rather
than into the per-level ``c_t`` values, so the per-level ``c_t``
spread is narrower than what an **anchored** parameterisation (e.g.
geometric-mean-1 constraint on ``log c_t``) would give. This is a
parameterisation choice, not a fitting failure: the predictive
content is identical, and the CAESER R-side analogue
``beta_pool_scale`` (which uses an anchor) reproduces the predictive
numbers of S-LC-ASH to within FP noise on the 246-level panel. If
interpretability of ``c_t`` as an absolute multiplier matters
downstream, an anchored variant is a natural follow-up.

Users diagnosing a single ``tau_c^2`` quote should therefore not
treat a small value as evidence of optimisation failure; check the
per-level ``c_t`` spread first. If ``c_t`` values are narrow and
predictives are good, the optimum is genuinely homogeneous and the
collapse is the right answer.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

from cebmf_torch.cebnm.lcash import lcash_PosteriorMeanNorm
from cebmf_torch.ebnm.ash import PriorType, ash

LOG2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Result container.
# ---------------------------------------------------------------------------


class slcash_PosteriorMeanNorm(lcash_PosteriorMeanNorm):
    """Result container for the S-LC-ASH family.

    Extends :class:`cebmf_torch.cebnm.lcash.lcash_PosteriorMeanNorm` with
    one extra field, :attr:`level_params`, carrying the per-level scalar
    ``c_t`` and shared spike weight ``p`` produced by the S-LC-ASH joint
    Adam fit (or by the cold-start L-BFGS fit for a single new level).

    The :meth:`predict_pi` method inherited from the LC-ASH parent does
    not apply here because S-LC-ASH does not parameterise the prior via
    a softmax/proportional-odds network of features. This subclass
    overrides :meth:`predict_pi` to raise :class:`NotImplementedError`
    with a pointer to the two correct scoring paths
    (:func:`s_lcash_new_level_posterior_means` for new levels;
    :func:`s_lcash_compute_posteriors` for known ``c_t``).

    Parameters
    ----------
    post_mean, post_mean2, post_sd, pi_np, scale, loss, model_param,
    priors_fitted, priors_fitted_history, marginal_loglik,
    x_means, x_stds, _arch_meta :
        See :class:`cebmf_torch.cebnm.lcash.lcash_PosteriorMeanNorm`.
        For S-LC-ASH, ``_arch_meta`` carries
        ``{"family": "s_lcash", "n_levels": int, "K": int}`` for the
        panel fit, or
        ``{"family": "s_lcash", "single_level": True, "K": int}`` for a
        cold-start result.
    level_params : dict or None, optional
        Per-level scalar parameters fitted by the S-LC-ASH family:
        ``{"c": Tensor (T,), "p": Tensor (1,)}`` where ``c`` is per
        level (one entry per level of the categorical covariate) and
        ``p`` is the shared spike weight.
    """

    def __init__(
        self,
        post_mean,
        post_mean2,
        post_sd,
        pi_np,
        scale,
        loss=0,
        model_param=None,
        *,
        priors_fitted=None,
        priors_fitted_history=None,
        marginal_loglik: float | None = None,
        x_means: torch.Tensor | None = None,
        x_stds: torch.Tensor | None = None,
        _arch_meta: dict | None = None,
        level_params: dict | None = None,
    ):
        super().__init__(
            post_mean=post_mean,
            post_mean2=post_mean2,
            post_sd=post_sd,
            pi_np=pi_np,
            scale=scale,
            loss=loss,
            model_param=model_param,
            priors_fitted=priors_fitted,
            priors_fitted_history=priors_fitted_history,
            marginal_loglik=marginal_loglik,
            x_means=x_means,
            x_stds=x_stds,
            _arch_meta=_arch_meta,
        )
        self.level_params = level_params

    def predict_pi(
        self,
        X: torch.Tensor | None = None,
        X_cat: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Not supported for the S-LC-ASH family.

        S-LC-ASH does not parameterise the prior via a softmax /
        proportional-odds network of features, so the inherited LC-ASH
        :meth:`predict_pi` does not apply.

        Two supported scoring paths:

        1. For genes from a NEW level (a level of the categorical
           covariate that was not present in the panel), call
           :func:`s_lcash_new_level_posterior_means` to fit a per-level
           ``c_t`` and obtain posteriors.
        2. For genes from a level that WAS in the panel (or any scoring
           with known ``c_t`` and shared ``p``), call
           :func:`s_lcash_compute_posteriors` directly with the per-level
           ``log_c``, the shared ``logit_p``, and the shared ``eta`` and
           ``sigma`` read off ``self.model_param``.
        """
        raise NotImplementedError(
            "predict_pi is not implemented for the S-LC-ASH family. "
            "Two supported paths to score new data: "
            "(1) for genes from a NEW level (a level of the categorical "
            "covariate that was not present in the panel), call "
            "cebmf_torch.cebnm.s_lcash_new_level_posterior_means(betahat, sebetahat, panel_result) "
            "to fit a per-level c_t and obtain posteriors via "
            "s_lcash_compute_posteriors; "
            "(2) for genes from a level that WAS in the panel (or any "
            "scoring with known c_t and shared p), call "
            "cebmf_torch.cebnm.s_lcash.s_lcash_compute_posteriors(...) "
            "directly with the per-level log_c, the shared logit_p, the "
            "shared eta and sigma read off panel_result.model_param."
        )

# ---------------------------------------------------------------------------
# Module-level defaults (single source of truth).
# ---------------------------------------------------------------------------
# Defaults that appear in more than one public function in this module
# (e.g. shared between :func:`s_lcash_posterior_means`, :func:`s_lcash_new_level_posterior_means`,
# and :func:`_warm_start_from_pooled_ash`) live here, as a single declaration,
# so they cannot drift across signatures.

#: Multiplicative step between adjacent slab-grid widths in the autoselect
#: grid. ``sqrt(2)`` matches the convention of R ``ashr``.
DEFAULT_GRID_MULT: float = math.sqrt(2.0)

#: Log-likelihood floor for the pooled-ash warm start.
DEFAULT_ASH_THRESHOLD: float = 1e-6

#: Whether the warm start uses ash's L-BFGS solver (``True``) or its EM
#: solver (``False``). L-BFGS gives sparser pi values; EM matches the
#: convention used in lcash's ``ash_init=False`` path.
DEFAULT_ASH_INIT: bool = True

#: Floor on the level-2 hyperparameter ``tau_c^2``. The joint-Adam trainer
#: applies this as a soft floor via ``exp(log_tau_c).clamp(min=sqrt(tau2_min))``.
DEFAULT_TAU2_MIN: float = 1e-6


# ---------------------------------------------------------------------------
# Kernels.
# ---------------------------------------------------------------------------


def s_lcash_log_marginal(
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    level_id: torch.Tensor,
    log_c: torch.Tensor,
    logit_p: torch.Tensor,
    eta: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Per-observation marginal log-density.

    .. math::

        \\log m_g \\;=\\; \\log\\!\\Big(
            p\\, \\mathcal{N}(\\hat\\beta_g\\mid 0, s_g^2)
            + (1 - p) \\sum_k w_k\\,
              \\mathcal{N}\\bigl(\\hat\\beta_g\\mid 0,\\, c_t^2\\sigma_k^2 + s_g^2\\bigr)
        \\Big)

    where ``t = level_id[g]``, ``c_t = exp(log_c[t])``, ``p = sigmoid(logit_p)``,
    ``w_k = softmax(eta)_k``. Computed via ``logsumexp`` over (K + 1) components.

    Parameters
    ----------
    betahat : Tensor, shape (N,)
        Observed effect estimates.
    sebetahat : Tensor, shape (N,)
        Standard errors. Strictly positive.
    level_id : Tensor, shape (N,), dtype long
        Per-observation level index in ``[0, T)``.
    log_c : Tensor, shape (T,)
        Per-level log-scale.
    logit_p : Tensor, scalar (0-d) or shape (1,)
        Shared spike weight on the logit scale.
    eta : Tensor, shape (K,)
        Pre-softmax shared slab weights.
    sigma : Tensor, shape (K,)
        Strictly positive shared slab widths.

    Returns
    -------
    log_m : Tensor, shape (N,)
        Per-observation marginal log-density.
    """
    if sigma.numel() == 0:
        raise ValueError("sigma must have at least one slab component (got K = 0).")
    if torch.any(sigma <= 0):
        raise ValueError(
            "sigma must contain only strictly positive slab widths. "
            "Strip the spike entry (sigma=0) before calling."
        )

    log_c_g = log_c.index_select(0, level_id)  # (N,)
    c_g = torch.exp(log_c_g)
    log_p = torch.nn.functional.logsigmoid(logit_p)
    log_1mp = torch.nn.functional.logsigmoid(-logit_p)
    log_w = torch.log_softmax(eta, dim=0)

    se2 = sebetahat * sebetahat
    sigma2 = sigma * sigma

    # Spike: N(beta | 0, s^2) weighted by p.
    log_spike = log_p - 0.5 * (LOG2PI + torch.log(se2) + (betahat * betahat) / se2)
    # Slab: K Normals N(0, c_t^2 * sigma_k^2 + s^2) weighted by (1-p)*w_k.
    var_slab = (c_g * c_g)[:, None] * sigma2[None, :] + se2[:, None]  # (N, K)
    log_slab = (
        log_1mp
        + log_w[None, :]
        - 0.5 * (LOG2PI + torch.log(var_slab) + (betahat[:, None] * betahat[:, None]) / var_slab)
    )  # (N, K)

    log_components = torch.cat([log_spike[:, None], log_slab], dim=1)  # (N, K+1)
    return torch.logsumexp(log_components, dim=1)


def s_lcash_compute_posteriors(
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    level_id: torch.Tensor,
    log_c: torch.Tensor,
    logit_p: torch.Tensor,
    eta: torch.Tensor,
    sigma: torch.Tensor,
) -> dict:
    """Per-observation posterior moments under the S-LC-ASH prior.

    Returns ``{"post_mean", "post_mean2", "post_sd", "pi_np", "log_marginal"}``.
    ``pi_np`` has shape (N, K + 1) with column 0 = spike responsibility,
    columns 1..K = slab responsibilities. Rows sum to 1.
    """
    if sigma.numel() == 0:
        raise ValueError("sigma must have at least one slab component (got K = 0).")
    if torch.any(sigma <= 0):
        raise ValueError("sigma must contain only strictly positive slab widths.")

    log_c_g = log_c.index_select(0, level_id)
    c_g = torch.exp(log_c_g)
    log_p = torch.nn.functional.logsigmoid(logit_p)
    log_1mp = torch.nn.functional.logsigmoid(-logit_p)
    log_w = torch.log_softmax(eta, dim=0)

    se2 = sebetahat * sebetahat
    sigma2 = sigma * sigma

    log_spike = log_p - 0.5 * (LOG2PI + torch.log(se2) + (betahat * betahat) / se2)
    c2 = c_g * c_g
    var_slab = c2[:, None] * sigma2[None, :] + se2[:, None]
    log_slab = (
        log_1mp
        + log_w[None, :]
        - 0.5 * (LOG2PI + torch.log(var_slab) + (betahat[:, None] * betahat[:, None]) / var_slab)
    )

    log_components = torch.cat([log_spike[:, None], log_slab], dim=1)
    log_marginal = torch.logsumexp(log_components, dim=1)
    pi_np = torch.exp(log_components - log_marginal[:, None])

    # Conditional posterior moments per slab component; spike contributes (m=0, v=0).
    ratio = (c2[:, None] * sigma2[None, :]) / var_slab
    m_slab = ratio * betahat[:, None]
    v_slab = ratio * se2[:, None]
    r_slab = pi_np[:, 1:]

    post_mean = (r_slab * m_slab).sum(dim=1)
    post_mean2 = (r_slab * (v_slab + m_slab * m_slab)).sum(dim=1)
    post_var = torch.clamp(post_mean2 - post_mean * post_mean, min=0.0)
    post_sd = torch.sqrt(post_var)

    return {
        "post_mean": post_mean,
        "post_mean2": post_mean2,
        "post_sd": post_sd,
        "pi_np": pi_np,
        "log_marginal": log_marginal,
    }


# ---------------------------------------------------------------------------
# Per-level + shared-parameter container.
# ---------------------------------------------------------------------------


class SLcashNet(nn.Module):
    """S-LC-ASH parameter container (single per-level dial, shared p).

    Trainable parameters:

    * ``log_c`` (T,) — per-level log-scale.
    * ``logit_p`` () scalar — shared spike weight on the logit scale.
    * ``eta`` (K,) — pre-softmax shared slab weights.
    * ``mu_c`` (), ``log_tau_c`` () — level-2 hyperparameters of the
      ``log c_t`` distribution; jointly optimised by the public trainer.

    Buffer:

    * ``sigma`` (K,) — strictly positive shared slab widths.
    """

    def __init__(
        self,
        n_levels: int,
        sigma: torch.Tensor,
        log_w_init: torch.Tensor,
        logit_p_init: float,
        log_c_init: float = 0.0,
        log_tau_c_init: float = 0.0,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        if n_levels < 1:
            raise ValueError(f"n_levels must be >= 1; got {n_levels}.")
        sigma_t = torch.as_tensor(sigma, dtype=dtype)
        if sigma_t.ndim != 1 or sigma_t.numel() < 1:
            raise ValueError(f"sigma must be 1-D with K >= 1; got shape {tuple(sigma_t.shape)}.")
        if torch.any(sigma_t <= 0):
            raise ValueError("sigma must contain only strictly positive slab widths.")
        log_w_init_t = torch.as_tensor(log_w_init, dtype=dtype)
        if log_w_init_t.shape != sigma_t.shape:
            raise ValueError(
                f"log_w_init shape {tuple(log_w_init_t.shape)} must match "
                f"sigma shape {tuple(sigma_t.shape)}."
            )

        self.n_levels = int(n_levels)
        self.K = int(sigma_t.numel())

        self.log_c = nn.Parameter(torch.full((n_levels,), float(log_c_init), dtype=dtype))
        self.logit_p = nn.Parameter(torch.tensor(float(logit_p_init), dtype=dtype))
        self.eta = nn.Parameter(log_w_init_t.clone())
        self.mu_c = nn.Parameter(torch.tensor(0.0, dtype=dtype))
        self.log_tau_c = nn.Parameter(torch.tensor(float(log_tau_c_init), dtype=dtype))

        self.register_buffer("sigma", sigma_t)

    def w(self) -> torch.Tensor:
        """Slab weights ``softmax(eta)``."""
        return torch.softmax(self.eta, dim=0)

    def c(self) -> torch.Tensor:
        """Per-level scale ``exp(log_c)``."""
        return torch.exp(self.log_c)

    def p(self) -> torch.Tensor:
        """Shared spike weight ``sigmoid(logit_p)``."""
        return torch.sigmoid(self.logit_p)

    def recentre_eta_(self) -> None:
        """In-place: subtract the mean of ``eta`` (preserves ``softmax(eta)``).

        Resolves the softmax translation gauge before saving state-dict.
        Call only outside of training.
        """
        with torch.no_grad():
            self.eta.sub_(self.eta.mean())


# ---------------------------------------------------------------------------
# Warm start from pooled ash.
# ---------------------------------------------------------------------------


def _warm_start_from_pooled_ash(
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    *,
    mult: float = DEFAULT_GRID_MULT,
    ash_threshold: float = DEFAULT_ASH_THRESHOLD,
    ash_init: bool = DEFAULT_ASH_INIT,
    initial_p_floor: float = 0.01,
    initial_p_ceiling: float = 0.99,
) -> dict:
    """Build a warm start from one pooled-ash fit.

    Returns ``{"sigma", "log_w", "logit_p_init", "p_pooled_raw"}`` with:

    * ``sigma`` — strictly positive slab widths (autoselect grid with the
      ``sigma=0`` entry stripped).
    * ``log_w`` — initial log-slab-weights (post-spike pi renormalised).
    * ``logit_p_init`` — pooled-ash spike weight, clipped to
      ``[initial_p_floor, initial_p_ceiling]``, on the logit scale.
    * ``p_pooled_raw`` — the unclipped pooled-ash spike weight, for
      diagnostics.
    """
    betahat = torch.as_tensor(betahat)
    sebetahat = torch.as_tensor(sebetahat)
    if betahat.shape != sebetahat.shape:
        raise ValueError(
            f"betahat and sebetahat must have the same shape; got "
            f"{tuple(betahat.shape)} and {tuple(sebetahat.shape)}."
        )

    optimizer = "lbfgs" if ash_init else "em"
    ash_result = ash(
        betahat,
        sebetahat,
        prior=PriorType.NORM,
        mult=mult,
        verbose=False,
        threshold_loglikelihood=math.log(max(ash_threshold, 1e-300)),
        optimizer=optimizer,
    )
    scale_full = ash_result.scale.detach()
    pi_full = ash_result.pi.detach() if ash_result.pi is not None else None
    if pi_full is None:
        raise RuntimeError("ash returned None for pi; cannot warm start.")
    if scale_full.numel() < 2 or float(scale_full[0]) != 0.0:
        raise RuntimeError(
            f"Expected ash to return scale[0] = 0 (spike) and at least one "
            f"slab component. Got scale[0]={float(scale_full[0])}, K_total="
            f"{scale_full.numel()}."
        )

    sigma = scale_full[1:].clone()
    p_pooled_raw = float(pi_full[0])
    p_pooled = max(initial_p_floor, min(initial_p_ceiling, p_pooled_raw))

    slab_pi = pi_full[1:].clone()
    slab_total = slab_pi.sum()
    if float(slab_total) <= 0.0:
        slab_pi = torch.full_like(slab_pi, 1.0 / max(slab_pi.numel(), 1))
    else:
        slab_pi = slab_pi / slab_total
    log_w = torch.log(torch.clamp(slab_pi, min=1e-300))

    logit_p_init = float(math.log(p_pooled / (1.0 - p_pooled)))
    return {
        "sigma": sigma,
        "log_w": log_w,
        "logit_p_init": logit_p_init,
        "p_pooled_raw": p_pooled_raw,
    }


# ---------------------------------------------------------------------------
# Public entry: panel fit.
#
# The training loop is inlined here, matching the convention of every other
# cEBNM solver in this package (e.g. :func:`cebmf_torch.cebnm.cash_posterior_means`,
# :func:`cebmf_torch.cebnm.emdn_posterior_means`). No internal worker mirrors
# this signature, so kwargs are declared exactly once.
# ---------------------------------------------------------------------------


def s_lcash_posterior_means(
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    X_cat: torch.Tensor,
    n_cat_levels: int,
    *,
    n_epochs: int = 500,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    tau2_min: float = DEFAULT_TAU2_MIN,
    mult: float = DEFAULT_GRID_MULT,
    ash_threshold: float = DEFAULT_ASH_THRESHOLD,
    ash_init: bool = DEFAULT_ASH_INIT,
    snapshot_every: int = 10,
    track_loglik_history: bool = False,
    device: torch.device | None = None,
    verbose: bool = True,
    seed: int = 42,
):
    """Fit S-LC-ASH on a multi-level panel.

    Minimises the negative log marginal joint

    .. math::

        \\mathrm{loss} = -\\sum_g \\log m_g
                       - \\sum_t \\log\\mathcal{N}\\!\\bigl(\\log c_t \\mid \\mu_c, \\tau_c^2\\bigr)

    over ``log_c`` (T,), ``logit_p`` (), ``eta`` (K,), ``mu_c`` (), and
    ``log_tau_c`` () jointly under a single Adam optimiser. The level-2
    quadratic term ``+sum_t (log c_t - mu_c)^2 / (2 tau_c^2)`` resists
    ``tau_c -> 0`` collapse when per-level ``log c_t`` values diverge
    from ``mu_c``; the ``+T*log(tau_c)`` normaliser drives ``tau_c``
    down, and the two balance at ``tau_c^2 = empirical variance``. See
    the module docstring for the panel-dependence caveats.

    Parameters
    ----------
    betahat : Tensor, shape (N,)
        Pooled effect estimates.
    sebetahat : Tensor, shape (N,)
        Pooled standard errors (strictly positive).
    X_cat : Tensor, shape (N,), dtype long
        Per-observation level index in ``[0, n_cat_levels)``.
    n_cat_levels : int
        Number of distinct levels in the panel.
    n_epochs : int, optional
        Adam steps (default 500).
    lr : float, optional
        Adam learning rate (default 1e-2).
    weight_decay : float, optional
        Adam weight-decay (default 0; the level-2 prior is the regulariser).
    tau2_min : float, optional
        Floor on ``tau_c^2`` (default :data:`DEFAULT_TAU2_MIN`). Applied
        as a soft floor via ``exp(log_tau_c).clamp(min=sqrt(tau2_min))``.
    mult, ash_threshold, ash_init : optional
        Forwarded to :func:`_warm_start_from_pooled_ash`. Defaults are the
        module-level constants :data:`DEFAULT_GRID_MULT`,
        :data:`DEFAULT_ASH_THRESHOLD`, :data:`DEFAULT_ASH_INIT`.
    snapshot_every : int, optional
        Snapshot the hyperparameters into ``priors_fitted_history`` every
        this many epochs (default 10).
    track_loglik_history : bool, optional
        Record per-epoch panel marginal log-likelihood (default False).
    device, verbose, seed : optional
        Standard kwargs. With ``verbose=True`` a short progress line is
        printed every 10 epochs.

    Returns
    -------
    slcash_PosteriorMeanNorm
        With these fields populated:

        * ``post_mean, post_mean2, post_sd`` (N,) per-observation moments.
        * ``pi_np`` (N, K+1) per-observation responsibilities; column 0 = spike.
        * ``scale`` (K,) slab widths.
        * ``model_param`` SLcashNet ``state_dict``.
        * ``priors_fitted = {0: {"mu_c", "tau2_c", "p", "solver": "s_lcash_joint_adam"}}``.
        * ``priors_fitted_history`` per-snapshot list, same shape.
        * ``marginal_loglik`` final panel log-likelihood under the prior.
        * ``level_params = {"c": Tensor (T,), "p": Tensor (1,)}``.
        * ``_arch_meta = {"family": "s_lcash", "n_levels": int, "K": int}``.
    """
    torch.manual_seed(seed)
    device = device or torch.device("cpu")
    betahat = torch.as_tensor(betahat, dtype=torch.float64).to(device)
    sebetahat = torch.as_tensor(sebetahat, dtype=torch.float64).to(device)
    X_cat = torch.as_tensor(X_cat, dtype=torch.long).to(device)

    if betahat.shape != sebetahat.shape:
        raise ValueError(
            f"betahat and sebetahat must have the same shape; got "
            f"{tuple(betahat.shape)} and {tuple(sebetahat.shape)}."
        )
    if X_cat.shape != betahat.shape:
        raise ValueError(
            f"X_cat must have shape {tuple(betahat.shape)}; got {tuple(X_cat.shape)}."
        )
    if torch.any(sebetahat <= 0):
        raise ValueError("sebetahat must be strictly positive.")
    if int(X_cat.max()) >= n_cat_levels or int(X_cat.min()) < 0:
        raise ValueError(
            f"X_cat values must lie in [0, {n_cat_levels}); got range "
            f"[{int(X_cat.min())}, {int(X_cat.max())}]."
        )

    # ---- Warm start (pooled-ASH fit gives slab grid + weights + pooled spike)
    warm = _warm_start_from_pooled_ash(
        betahat, sebetahat, mult=mult, ash_threshold=ash_threshold, ash_init=ash_init
    )
    sigma = warm["sigma"].to(device)
    log_w = warm["log_w"].to(device)

    model = SLcashNet(
        n_levels=n_cat_levels,
        sigma=sigma,
        log_w_init=log_w,
        logit_p_init=warm["logit_p_init"],
        log_c_init=0.0,
        dtype=torch.float64,
    ).to(device)

    # ---- Joint-Adam training loop (inlined)
    optimizer = torch.optim.Adam(
        [model.log_c, model.logit_p, model.eta, model.mu_c, model.log_tau_c],
        lr=lr,
        weight_decay=weight_decay,
    )
    # Smooth floor on tau_c: tau_c = sqrt(tau2_min + exp(2 * log_tau_c)).
    # At all values of log_tau_c the gradient flows; for very negative
    # log_tau_c the formula asymptotes to sqrt(tau2_min) (the floor).
    # A hard `clamp(min=...)` would zero the gradient through tau_c
    # below the floor and prevent recovery if per-level heterogeneity
    # later emerges.
    tau2_min_t = torch.tensor(max(tau2_min, 1e-300), dtype=torch.float64, device=device)
    history: list[dict] = []
    loglik_history: list[float] = []
    final_loss = float("nan")
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        log_m = s_lcash_log_marginal(
            betahat, sebetahat, X_cat, model.log_c, model.logit_p, model.eta, model.sigma
        )
        data_loss = -log_m.sum()
        # Level-2 prior: log c_t ~ N(mu_c, tau_c^2).
        # Negative log-prob summed over T levels =
        #   +T * log(tau_c)                    (drives tau_c down)
        # + sum_t (log c_t - mu_c)^2 / (2 tau_c^2)  (resists tau_c -> 0
        #                                          when log_c_t differs
        #                                          from mu_c)
        # plus a constant 0.5 * T * log(2*pi).
        tau_c = torch.sqrt(tau2_min_t + torch.exp(2.0 * model.log_tau_c))
        pen = -Normal(loc=model.mu_c, scale=tau_c).log_prob(model.log_c).sum()
        loss = data_loss + pen
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

        if track_loglik_history:
            with torch.no_grad():
                loglik_history.append(float(log_m.sum().item()))
        if (epoch + 1) % snapshot_every == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                tau_c_eff = float(torch.sqrt(tau2_min_t + torch.exp(2.0 * model.log_tau_c)))
                history.append(
                    {
                        "mu_c": float(model.mu_c),
                        "tau2_c": tau_c_eff ** 2,
                        "p": float(torch.sigmoid(model.logit_p)),
                        "epoch": int(epoch + 1),
                    }
                )
        if verbose and (epoch + 1) % 10 == 0:
            print(f"[S-LC-ASH] Epoch {epoch + 1}/{n_epochs} | Loss: {final_loss:.4f}")

    # ---- Resolve softmax translation gauge before saving state_dict
    model.recentre_eta_()

    # ---- Final hyperparameters
    with torch.no_grad():
        tau_c_eff = float(torch.sqrt(tau2_min_t + torch.exp(2.0 * model.log_tau_c)))
    psi = {
        "mu_c": float(model.mu_c),
        "tau2_c": tau_c_eff ** 2,
        "p": float(torch.sigmoid(model.logit_p)),
        "solver": "s_lcash_joint_adam",
    }

    # ---- Posteriors on the panel data using the fitted prior
    with torch.no_grad():
        out = s_lcash_compute_posteriors(
            betahat, sebetahat, X_cat, model.log_c, model.logit_p, model.eta, model.sigma
        )
    marginal_loglik = float(out["log_marginal"].sum().item())

    priors_fitted = {0: dict(psi)}
    history_wrapped: list[dict] = [
        {0: dict(snap, solver="s_lcash_joint_adam")} for snap in history
    ]
    if track_loglik_history and history_wrapped:
        history_wrapped[-1][0]["loglik_history"] = list(loglik_history)
    elif track_loglik_history:
        history_wrapped = [
            {0: {"loglik_history": list(loglik_history), "solver": "s_lcash_joint_adam"}}
        ]

    arch_meta = {"family": "s_lcash", "n_levels": int(n_cat_levels), "K": int(sigma.numel())}
    level_params = {
        "c": model.c().detach().cpu(),
        "p": torch.sigmoid(model.logit_p).detach().cpu().reshape(1),
    }

    return slcash_PosteriorMeanNorm(
        post_mean=out["post_mean"].detach().cpu(),
        post_mean2=out["post_mean2"].detach().cpu(),
        post_sd=out["post_sd"].detach().cpu(),
        pi_np=out["pi_np"].detach().cpu(),
        scale=model.sigma.detach().cpu(),
        loss=final_loss,
        model_param=model.state_dict(),
        priors_fitted=priors_fitted,
        priors_fitted_history=history_wrapped,
        marginal_loglik=marginal_loglik,
        _arch_meta=arch_meta,
        level_params=level_params,
    )


# ---------------------------------------------------------------------------
# Public entry: cold-start.
# ---------------------------------------------------------------------------


def s_lcash_new_level_posterior_means(
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    panel_result,
    *,
    n_epochs: int = 200,
    lr: float = 1.0,
    tau2_min: float = DEFAULT_TAU2_MIN,
    tau_inflate: float = 1.0,
    track_loglik_history: bool = False,
    device: torch.device | None = None,
    verbose: bool = False,
    seed: int = 42,
):
    """Cold-start a single new level given a panel-trained S-LC-ASH model.

    Solves a one-parameter MAP problem for ``log c_new`` with the panel's
    Layer-B parameters (``mu_c``, ``tau_c^2``, ``logit_p``, ``eta``, ``sigma``)
    held fixed.

    .. math::

        \\hat u_{*} \\;=\\; \\arg\\max_u\\;
            \\sum_g \\log m\\bigl(\\hat\\beta_g\\mid s_g,\\, e^u; \\hat\\Theta\\bigr)
            \\;+\\; \\log\\mathcal{N}(u\\mid \\hat\\mu_c, \\rho^2 \\hat\\tau_c^2)

    where ``rho = tau_inflate``. Solved with PyTorch's L-BFGS (strong-Wolfe
    line search). Converges in a handful of outer iterations on smooth data.

    Parameters
    ----------
    betahat, sebetahat : Tensor, shape (n_t,)
        New level's data (unbatched).
    panel_result : slcash_PosteriorMeanNorm
        Output of :func:`s_lcash_posterior_means`.
    n_epochs : int, optional
        Hard cap on the optimisation budget. Default 200.
    lr : float, optional
        Initial step size for L-BFGS strong-Wolfe (default 1.0).
    tau2_min : float, optional
        Floor on ``tau_c^2`` when used in the prior (default 1e-6).
    tau_inflate : float, optional
        Multiplicative inflation on the deployment-time prior spread
        ``tau_c`` (default 1.0). Larger values widen the prior on a new
        level when the panel-fitted ``tau_c`` is too tight to represent
        deployment-time variability.
    track_loglik_history : bool, optional
        Record per-closure-call data marginal log-likelihood. With
        L-BFGS strong-Wolfe line search the closure runs multiple times
        per outer iteration (each line-search probe), so the history is
        indexed by closure call (line-search probe), NOT by outer
        iteration or "epoch". Granularity is finer than the
        :func:`s_lcash_posterior_means` history which is one entry per
        Adam epoch.
    device, verbose, seed : standard kwargs.

    Returns
    -------
    slcash_PosteriorMeanNorm
        Per-observation posteriors plus per-level ``c`` (1-element tensor).
        ``_arch_meta = {"family": "s_lcash", "single_level": True, "K": K}``.
    """

    arch_meta = getattr(panel_result, "_arch_meta", None)
    if arch_meta is None or arch_meta.get("family") != "s_lcash":
        raise ValueError(
            "s_lcash_new_level_posterior_means requires a panel_result produced by "
            "s_lcash_posterior_means; got _arch_meta={!r}.".format(arch_meta)
        )
    panel_priors = getattr(panel_result, "priors_fitted", None)
    if not panel_priors or 0 not in panel_priors:
        raise ValueError("panel_result.priors_fitted is missing the level-2 hyperparameters.")
    psi = panel_priors[0]
    mu_c = float(psi["mu_c"])
    tau2_c = max(float(psi["tau2_c"]), tau2_min) * (tau_inflate ** 2)
    tau_c_eff = math.sqrt(tau2_c)

    state = panel_result.model_param
    if state is None:
        raise ValueError("panel_result.model_param is None; cannot reconstruct the panel.")
    device = device or torch.device("cpu")
    eta = state["eta"].detach().to(device).clone()
    sigma = state["sigma"].detach().to(device).clone()
    logit_p = state["logit_p"].detach().to(device).clone()

    torch.manual_seed(seed)
    betahat = torch.as_tensor(betahat, dtype=torch.float64).to(device)
    sebetahat = torch.as_tensor(sebetahat, dtype=torch.float64).to(device)
    if betahat.shape != sebetahat.shape:
        raise ValueError(
            "betahat and sebetahat must have the same shape; got "
            f"{tuple(betahat.shape)} and {tuple(sebetahat.shape)}."
        )
    if torch.any(sebetahat <= 0):
        raise ValueError("sebetahat must be strictly positive.")
    n_new = betahat.numel()
    level_id = torch.zeros(n_new, dtype=torch.long, device=device)

    log_c = torch.tensor([mu_c], dtype=torch.float64, device=device, requires_grad=True)

    optimizer = torch.optim.LBFGS(
        [log_c], lr=lr, max_iter=20,
        tolerance_grad=1e-7, tolerance_change=1e-9, history_size=10,
        line_search_fn="strong_wolfe",
    )
    hyper = Normal(loc=torch.tensor(mu_c, dtype=torch.float64, device=device),
                    scale=torch.tensor(tau_c_eff, dtype=torch.float64, device=device))
    loglik_history: list[float] = []

    def closure():
        optimizer.zero_grad()
        log_m = s_lcash_log_marginal(
            betahat, sebetahat, level_id, log_c, logit_p, eta, sigma
        )
        data_loss = -log_m.sum()
        pen = -hyper.log_prob(log_c[0])
        loss = data_loss + pen
        loss.backward()
        if track_loglik_history:
            with torch.no_grad():
                loglik_history.append(float(log_m.sum().item()))
        return loss

    # `final_loss` records the joint MAP objective at the optimum
    # (data + prior penalty), matching the convention used by
    # :func:`s_lcash_posterior_means`. We deliberately do NOT store
    # `-marginal_loglik` here because that excludes the prior penalty
    # and would be inconsistent with how the panel-fit `loss` field
    # is interpreted.
    prev_loss = None
    final_loss = float("nan")
    for _ in range(max(1, n_epochs // 20)):
        loss = optimizer.step(closure)
        cur = float(loss.item())
        final_loss = cur
        if prev_loss is not None and abs(prev_loss - cur) < 1e-9:
            break
        prev_loss = cur

    with torch.no_grad():
        out = s_lcash_compute_posteriors(
            betahat, sebetahat, level_id, log_c, logit_p, eta, sigma
        )
    marginal_loglik = float(out["log_marginal"].sum().item())

    if verbose:
        print(
            f"[s_lcash_new_level_posterior_means] c={float(torch.exp(log_c[0])):.4g} "
            f"p={float(torch.sigmoid(logit_p)):.4g} "
            f"marginal_loglik={marginal_loglik:.4g}"
        )

    arch_meta_out = {"family": "s_lcash", "single_level": True, "K": int(sigma.numel())}
    c_val = torch.exp(log_c.detach()).cpu()
    p_val = torch.sigmoid(logit_p.detach()).cpu().reshape(1)
    level_params = {"c": c_val, "p": p_val}
    priors_fitted = {
        0: {
            "mu_c": mu_c,
            "tau2_c": tau2_c,
            "p": float(p_val[0]),
            "solver": "s_lcash_cold_start",
            "frozen_from_panel": True,
            "tau_inflate": float(tau_inflate),
        }
    }
    history_wrapped: list[dict] = []
    if track_loglik_history:
        history_wrapped = [{0: {"loglik_history": loglik_history,
                                  "solver": "s_lcash_cold_start"}}]

    return slcash_PosteriorMeanNorm(
        post_mean=out["post_mean"].detach().cpu(),
        post_mean2=out["post_mean2"].detach().cpu(),
        post_sd=out["post_sd"].detach().cpu(),
        pi_np=out["pi_np"].detach().cpu(),
        scale=sigma.detach().cpu(),
        loss=final_loss,
        model_param={
            "log_c": log_c.detach().cpu(),
            "logit_p": logit_p.detach().cpu(),
            "eta": eta.detach().cpu(),
            "sigma": sigma.detach().cpu(),
        },
        priors_fitted=priors_fitted,
        priors_fitted_history=history_wrapped,
        marginal_loglik=marginal_loglik,
        _arch_meta=arch_meta_out,
        level_params=level_params,
    )


__all__ = [
    "SLcashNet",
    "s_lcash_compute_posteriors",
    "s_lcash_log_marginal",
    "s_lcash_new_level_posterior_means",
    "s_lcash_posterior_means",
]
