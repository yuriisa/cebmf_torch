"""S-LC-ASH: Scaled Linear Covariate-mediated Adaptive Shrinkage.

Per-trait spike-and-slab where the slab scale ``c_t`` is per-trait and
everything else is shared:

.. math::

    \\beta_{g,t} \\sim p \\cdot \\delta_0 + (1 - p) \\sum_{k=1}^{K} w_k\\,
    \\mathcal{N}\\!\\left(0,\\ c_t^2\\sigma_k^2\\right)

* ``p`` (scalar) — shared spike weight (fraction of null genes, panel-wide).
* ``w_k``, ``sigma_k`` — shared slab weights and widths.
* ``c_t > 0`` — per-trait scale on the slab widths. Hyperprior
  ``log c_t ~ N(mu_c, tau_c^2)`` with ``mu_c`` and ``tau_c^2`` empirical-Bayes
  fitted (jointly with everything else by Adam, by default).

The model is a one-dial per-trait extension of pooled `ash` and matches the
sister-project ``beta_pool_scale`` parameterisation. The "LC" in the name
keeps the door open for covariate dependence on ``c_t`` in a future PR
(e.g. phenotype-class regulating ``log c`` linearly); the present
implementation has a single categorical covariate (trait id).

Public API:

* :func:`s_lc_ash_posterior_means` — panel fit.
* :func:`fit_new_trait` — cold-start a new trait given a panel-trained model.
* :class:`SLCAshNet` — per-trait + shared-parameter container (``nn.Module``).
* :func:`s_lc_ash_log_marginal` — per-observation marginal log-density kernel.
* :func:`s_lc_ash_compute_posteriors` — per-observation posterior moments kernel.
* :func:`warm_start_from_pooled_ash` — slab + pooled-spike warm-start helper.

Notes on the level-2 hyperparameter ``tau_c^2``
-----------------------------------------------

The level-2 prior on ``log c_t`` is a free-mean Normal
``N(mu_c, tau_c^2)`` with ``mu_c`` and ``log tau_c`` learnable
parameters optimised jointly with everything else under a single
Adam loop. The Normal log-density's ``-T*log(tau_c)`` normaliser
provides a restoring gradient against ``tau_c -> 0`` that closed-form
alternating empirical Bayes lacks. **However, this restoring force
acts only when at least some per-trait ``log c_t`` values diverge from
``mu_c``.** On panels where the data prefers a homogeneous solution
(every ``c_t`` close to the panel mean), the level-2 quadratic term
collapses to zero, the ``-T*log(tau_c)`` term dominates, and
``tau_c^2`` shrinks to the ``tau2_min`` floor.

This was observed by the CAESER 246-trait validation in both
training directions: the per-trait ``c_t`` values clustered tightly
around the panel mean (range ``[0.949, 0.966]`` on UKB-trained,
``[0.984, 0.986]`` on AGD-trained), and ``tau_c^2`` collapsed to
``~10^-6`` over a few hundred epochs. The marginal log-likelihood
decreased by tens of nats panel-total over a 1500-epoch fit,
matching the diagnostic signature of misspecified-prior collapse,
**but the predictive impact is negligible** (paired test LPD changes
by ``~10^-5``). The model degrades gracefully to "pooled ASH plus
shared spike" in this regime, which itself beats per-trait ASH on
that panel.

The deployment workflow side-steps the issue by panel construction:
restricting the panel to well-pinned high-power traits (and adding
underpowered traits via :func:`fit_new_trait`) keeps the per-trait
``c_t`` values diverse enough that ``tau_c^2`` stays well above the
floor. CAESER measured ``tau_c^2 = 0.115`` on the well-pinned 50-trait
subset versus ``4 x 10^-4`` on the full 246-trait fit.

A consequence of the **free-mean** parameterisation: any panel-wide
multiplicative shift in slab width is absorbed into ``mu_c`` rather
than into the per-trait ``c_t`` values, so the per-trait ``c_t``
spread is narrower than what an **anchored** parameterisation (e.g.
geometric-mean-1 constraint on ``log c_t``) would give. This is a
parameterisation choice, not a fitting failure: the predictive
content is identical, and the CAESER R-side analogue
``beta_pool_scale`` (which uses an anchor) reproduces the predictive
numbers of S-LC-ASH to within FP noise on the 246-trait panel. If
interpretability of ``c_t`` as an absolute multiplier matters
downstream, an anchored variant is a natural follow-up.

Users diagnosing a single ``tau_c^2`` quote should therefore not
treat a small value as evidence of optimisation failure; check the
per-trait ``c_t`` spread first. If ``c_t`` values are narrow and
predictives are good, the optimum is genuinely homogeneous and the
collapse is the right answer.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

from cebmf_torch.ebnm.ash import PriorType, ash

LOG2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Kernels.
# ---------------------------------------------------------------------------


def s_lc_ash_log_marginal(
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    trait_id: torch.Tensor,
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

    where ``t = trait_id[g]``, ``c_t = exp(log_c[t])``, ``p = sigmoid(logit_p)``,
    ``w_k = softmax(eta)_k``. Computed via ``logsumexp`` over (K + 1) components.

    Parameters
    ----------
    betahat : Tensor, shape (N,)
        Observed effect estimates.
    sebetahat : Tensor, shape (N,)
        Standard errors. Strictly positive.
    trait_id : Tensor, shape (N,), dtype long
        Per-observation trait index in ``[0, T)``.
    log_c : Tensor, shape (T,)
        Per-trait log-scale.
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

    log_c_g = log_c.index_select(0, trait_id)  # (N,)
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


def s_lc_ash_compute_posteriors(
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    trait_id: torch.Tensor,
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

    log_c_g = log_c.index_select(0, trait_id)
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
# Per-trait + shared-parameter container.
# ---------------------------------------------------------------------------


class SLCAshNet(nn.Module):
    """S-LC-ASH parameter container (single per-trait dial, shared p).

    Trainable parameters:

    * ``log_c`` (T,) — per-trait log-scale.
    * ``logit_p`` () scalar — shared spike weight on the logit scale.
    * ``eta`` (K,) — pre-softmax shared slab weights.
    * ``mu_c`` (), ``log_tau_c`` () — only when ``learnable_hyperparams=True``;
      the level-2 hyperparameters of the ``log c_t`` distribution.

    Buffer:

    * ``sigma`` (K,) — strictly positive shared slab widths.
    """

    def __init__(
        self,
        n_traits: int,
        sigma: torch.Tensor,
        log_w_init: torch.Tensor,
        logit_p_init: float,
        log_c_init: float = 0.0,
        log_c_init_spread: float = 0.0,
        seed: int = 42,
        dtype: torch.dtype = torch.float64,
        learnable_hyperparams: bool = True,
        log_tau_c_init: float = 0.0,
    ):
        super().__init__()
        if n_traits < 1:
            raise ValueError(f"n_traits must be >= 1; got {n_traits}.")
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

        self.n_traits = int(n_traits)
        self.K = int(sigma_t.numel())
        self.learnable_hyperparams = bool(learnable_hyperparams)

        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        if log_c_init_spread > 0.0:
            log_c_noise = torch.randn(n_traits, generator=gen, dtype=dtype) * float(log_c_init_spread)
        else:
            log_c_noise = torch.zeros(n_traits, dtype=dtype)

        self.log_c = nn.Parameter(torch.full((n_traits,), float(log_c_init), dtype=dtype) + log_c_noise)
        self.logit_p = nn.Parameter(torch.tensor(float(logit_p_init), dtype=dtype))
        self.eta = nn.Parameter(log_w_init_t.clone())

        if self.learnable_hyperparams:
            self.mu_c = nn.Parameter(torch.tensor(0.0, dtype=dtype))
            self.log_tau_c = nn.Parameter(torch.tensor(float(log_tau_c_init), dtype=dtype))

        self.register_buffer("sigma", sigma_t)

    def w(self) -> torch.Tensor:
        """Slab weights ``softmax(eta)``."""
        return torch.softmax(self.eta, dim=0)

    def c(self) -> torch.Tensor:
        """Per-trait scale ``exp(log_c)``."""
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


def warm_start_from_pooled_ash(
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    *,
    mult: float = math.sqrt(2.0),
    ash_threshold: float = 1e-6,
    ash_init: bool = True,
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
# Joint-Adam trainer (default).
# ---------------------------------------------------------------------------


def _train_joint(
    model: SLCAshNet,
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    trait_id: torch.Tensor,
    *,
    n_epochs: int,
    lr: float,
    weight_decay: float,
    tau2_min: float,
    track_loglik_history: bool,
    verbose: bool,
    snapshot_every: int = 10,
) -> dict:
    """Joint-Adam minimisation of the negative log marginal joint:

    .. math::

        \\mathrm{loss} = -\\sum_g \\log m_g - \\sum_t \\log\\mathcal{N}(\\log c_t \\mid \\mu_c, \\tau_c^2)

    The ``-log Normal`` term includes the ``log tau_c`` constant required when
    ``tau_c`` is itself a learnable parameter; this is what stabilises ``tau_c``
    against degenerate collapse to zero.
    """
    if not model.learnable_hyperparams:
        raise ValueError("_train_joint requires learnable_hyperparams=True.")
    optimizer = torch.optim.Adam(
        [model.log_c, model.logit_p, model.eta, model.mu_c, model.log_tau_c],
        lr=lr,
        weight_decay=weight_decay,
    )
    tau_floor = math.sqrt(max(tau2_min, 1e-300))

    history: list[dict] = []
    loglik_history: list[float] = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        log_m = s_lc_ash_log_marginal(
            betahat, sebetahat, trait_id, model.log_c, model.logit_p, model.eta, model.sigma
        )
        data_loss = -log_m.sum()
        tau_c = torch.exp(model.log_tau_c).clamp(min=tau_floor)
        # Use torch.distributions for the level-2 prior.
        hyper = Normal(loc=model.mu_c, scale=tau_c)
        pen = -hyper.log_prob(model.log_c).sum()
        loss = data_loss + pen
        loss.backward()
        optimizer.step()

        if track_loglik_history:
            with torch.no_grad():
                loglik_history.append(float(log_m.sum().item()))
        if (epoch + 1) % snapshot_every == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                tau_c_eff = float(torch.exp(model.log_tau_c).clamp(min=tau_floor))
                history.append(
                    {
                        "mu_c": float(model.mu_c),
                        "tau2_c": tau_c_eff ** 2,
                        "p": float(torch.sigmoid(model.logit_p)),
                        "epoch": int(epoch + 1),
                    }
                )

    with torch.no_grad():
        tau_c_eff = float(torch.exp(model.log_tau_c).clamp(min=tau_floor))
    psi = {
        "mu_c": float(model.mu_c),
        "tau2_c": tau_c_eff ** 2,
        "p": float(torch.sigmoid(model.logit_p)),
        "solver": "s_lc_ash_joint_adam",
    }

    model.recentre_eta_()
    return {
        "psi": psi,
        "history": history,
        "loglik_history": loglik_history,
        "final_loss": float(loss.item()),
        "epochs_done": n_epochs,
    }


# ---------------------------------------------------------------------------
# Public entry: panel fit.
# ---------------------------------------------------------------------------


def s_lc_ash_posterior_means(
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    X_cat: torch.Tensor,
    n_cat_levels: int,
    *,
    n_epochs: int = 500,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    tau2_min: float = 1e-6,
    log_c_init_spread: float = 0.0,
    mult: float = math.sqrt(2.0),
    ash_threshold: float = 1e-6,
    ash_init: bool = True,
    snapshot_every: int = 10,
    track_loglik_history: bool = False,
    device: torch.device | None = None,
    verbose: bool = True,
    seed: int = 42,
):
    """Fit S-LC-ASH on a multi-trait panel.

    Parameters
    ----------
    betahat : Tensor, shape (N,)
        Pooled effect estimates.
    sebetahat : Tensor, shape (N,)
        Pooled standard errors (strictly positive).
    X_cat : Tensor, shape (N,), dtype long
        Per-observation trait index in ``[0, n_cat_levels)``.
    n_cat_levels : int
        Number of distinct traits in the panel.
    n_epochs : int, optional
        Adam steps (default 500).
    lr : float, optional
        Adam learning rate (default 1e-2).
    weight_decay : float, optional
        Adam weight-decay (default 0; the level-2 prior is the regulariser).
    tau2_min : float, optional
        Floor on ``tau_c^2`` (default 1e-6).
    log_c_init_spread : float, optional
        Optional Gaussian spread on the per-trait ``log c`` initial values.
        Default 0 = identical init across traits.
    mult, ash_threshold, ash_init : optional
        Forwarded to :func:`warm_start_from_pooled_ash`.
    snapshot_every : int, optional
        Snapshot the hyperparameters into ``priors_fitted_history`` every
        this many epochs (default 10).
    track_loglik_history : bool, optional
        Record per-epoch panel marginal log-likelihood (default False).
    device, verbose, seed : optional
        Standard kwargs.

    Returns
    -------
    cash_PosteriorMeanNorm
        With these fields populated:

        * ``post_mean, post_mean2, post_sd`` (N,) per-observation moments.
        * ``pi_np`` (N, K+1) per-observation responsibilities; column 0 = spike.
        * ``scale`` (K,) slab widths.
        * ``model_param`` SLCAshNet ``state_dict``.
        * ``priors_fitted = {0: {"mu_c", "tau2_c", "p", "solver": "s_lc_ash_joint_adam"}}``.
        * ``priors_fitted_history`` per-snapshot list, same shape.
        * ``marginal_loglik`` final panel log-likelihood under the prior.
        * ``trait_params = {"c": Tensor (T,), "p": Tensor (1,)}``.
        * ``_arch_meta = {"family": "s_lc_ash", "T": int, "K": int}``.
    """
    from cebmf_torch.cebnm.cash_solver import cash_PosteriorMeanNorm

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

    warm = warm_start_from_pooled_ash(
        betahat, sebetahat, mult=mult, ash_threshold=ash_threshold, ash_init=ash_init
    )
    sigma = warm["sigma"].to(device)
    log_w = warm["log_w"].to(device)

    model = SLCAshNet(
        n_traits=n_cat_levels,
        sigma=sigma,
        log_w_init=log_w,
        logit_p_init=warm["logit_p_init"],
        log_c_init=0.0,
        log_c_init_spread=log_c_init_spread,
        seed=seed,
        dtype=torch.float64,
        learnable_hyperparams=True,
    ).to(device)

    train_out = _train_joint(
        model, betahat, sebetahat, X_cat,
        n_epochs=n_epochs, lr=lr, weight_decay=weight_decay,
        tau2_min=tau2_min,
        track_loglik_history=track_loglik_history,
        verbose=verbose,
        snapshot_every=snapshot_every,
    )

    with torch.no_grad():
        out = s_lc_ash_compute_posteriors(
            betahat, sebetahat, X_cat, model.log_c, model.logit_p, model.eta, model.sigma
        )

    marginal_loglik = float(out["log_marginal"].sum().item())

    psi = train_out["psi"]
    priors_fitted = {0: dict(psi)}
    history_wrapped: list[dict] = [{0: dict(snap, solver="s_lc_ash_joint_adam")}
                                    for snap in train_out["history"]]
    if track_loglik_history and history_wrapped:
        history_wrapped[-1][0]["loglik_history"] = list(train_out["loglik_history"])
    elif track_loglik_history:
        history_wrapped = [{0: {"loglik_history": list(train_out["loglik_history"]),
                                  "solver": "s_lc_ash_joint_adam"}}]

    arch_meta = {"family": "s_lc_ash", "T": int(n_cat_levels), "K": int(sigma.numel())}
    trait_params = {
        "c": model.c().detach().cpu(),
        "p": torch.sigmoid(model.logit_p).detach().cpu().reshape(1),
    }

    return cash_PosteriorMeanNorm(
        post_mean=out["post_mean"].detach().cpu(),
        post_mean2=out["post_mean2"].detach().cpu(),
        post_sd=out["post_sd"].detach().cpu(),
        pi_np=out["pi_np"].detach().cpu(),
        scale=model.sigma.detach().cpu(),
        loss=train_out["final_loss"],
        model_param=model.state_dict(),
        priors_fitted=priors_fitted,
        priors_fitted_history=history_wrapped,
        marginal_loglik=marginal_loglik,
        _arch_meta=arch_meta,
        trait_params=trait_params,
    )


# ---------------------------------------------------------------------------
# Public entry: cold-start.
# ---------------------------------------------------------------------------


def fit_new_trait(
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    panel_result,
    *,
    n_epochs: int = 200,
    lr: float = 1.0,
    tau2_min: float = 1e-6,
    tau_inflate: float = 1.0,
    track_loglik_history: bool = False,
    device: torch.device | None = None,
    verbose: bool = False,
    seed: int = 42,
):
    """Cold-start a single new trait given a panel-trained S-LC-ASH model.

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
        New trait's data (unbatched).
    panel_result : cash_PosteriorMeanNorm
        Output of :func:`s_lc_ash_posterior_means`.
    n_epochs : int, optional
        Hard cap on the optimisation budget. Default 200.
    lr : float, optional
        Initial step size for L-BFGS strong-Wolfe (default 1.0).
    tau2_min : float, optional
        Floor on ``tau_c^2`` when used in the prior (default 1e-6).
    tau_inflate : float, optional
        Multiplicative inflation on the deployment-time prior spread
        ``tau_c`` (default 1.0). Larger values widen the prior on a new
        trait when the panel-fitted ``tau_c`` is too tight to represent
        deployment-time variability.
    track_loglik_history, device, verbose, seed : standard kwargs.

    Returns
    -------
    cash_PosteriorMeanNorm
        Per-observation posteriors plus per-trait ``c`` (1-element tensor).
        ``_arch_meta = {"family": "s_lc_ash", "single_trait": True, "K": K}``.
    """
    from cebmf_torch.cebnm.cash_solver import cash_PosteriorMeanNorm

    arch_meta = getattr(panel_result, "_arch_meta", None)
    if arch_meta is None or arch_meta.get("family") != "s_lc_ash":
        raise ValueError(
            "fit_new_trait requires a panel_result produced by "
            "s_lc_ash_posterior_means; got _arch_meta={!r}.".format(arch_meta)
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
    trait_id = torch.zeros(n_new, dtype=torch.long, device=device)

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
        log_m = s_lc_ash_log_marginal(
            betahat, sebetahat, trait_id, log_c, logit_p, eta, sigma
        )
        data_loss = -log_m.sum()
        pen = -hyper.log_prob(log_c[0])
        loss = data_loss + pen
        loss.backward()
        if track_loglik_history:
            with torch.no_grad():
                loglik_history.append(float(log_m.sum().item()))
        return loss

    prev_loss = None
    for _ in range(max(1, n_epochs // 20)):
        loss = optimizer.step(closure)
        cur = float(loss.item())
        if prev_loss is not None and abs(prev_loss - cur) < 1e-9:
            break
        prev_loss = cur

    with torch.no_grad():
        out = s_lc_ash_compute_posteriors(
            betahat, sebetahat, trait_id, log_c, logit_p, eta, sigma
        )
    marginal_loglik = float(out["log_marginal"].sum().item())

    if verbose:
        print(
            f"[fit_new_trait] c={float(torch.exp(log_c[0])):.4g} "
            f"p={float(torch.sigmoid(logit_p)):.4g} "
            f"marginal_loglik={marginal_loglik:.4g}"
        )

    arch_meta_out = {"family": "s_lc_ash", "single_trait": True, "K": int(sigma.numel())}
    c_val = torch.exp(log_c.detach()).cpu()
    p_val = torch.sigmoid(logit_p.detach()).cpu().reshape(1)
    trait_params = {"c": c_val, "p": p_val}
    priors_fitted = {
        0: {
            "mu_c": mu_c,
            "tau2_c": tau2_c,
            "p": float(p_val[0]),
            "solver": "s_lc_ash_cold_start",
            "frozen_from_panel": True,
            "tau_inflate": float(tau_inflate),
        }
    }
    history_wrapped: list[dict] = []
    if track_loglik_history:
        history_wrapped = [{0: {"loglik_history": loglik_history,
                                  "solver": "s_lc_ash_cold_start"}}]

    return cash_PosteriorMeanNorm(
        post_mean=out["post_mean"].detach().cpu(),
        post_mean2=out["post_mean2"].detach().cpu(),
        post_sd=out["post_sd"].detach().cpu(),
        pi_np=out["pi_np"].detach().cpu(),
        scale=sigma.detach().cpu(),
        loss=-marginal_loglik,
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
        trait_params=trait_params,
    )


__all__ = [
    "SLCAshNet",
    "s_lc_ash_log_marginal",
    "s_lc_ash_compute_posteriors",
    "warm_start_from_pooled_ash",
    "s_lc_ash_posterior_means",
    "fit_new_trait",
]
