"""LC-ASH: Linear Covariate Adaptive Shrinkage.

Two parameterisations:
  - Softmax (multinomial logistic): K independent logit vectors, K*F params.
  - Proportional odds (ordered logistic): shared weight vector, F+K-1 params.

Both map gene features to mixture weights.  A linear alternative to the
MLP-based CASH solver, with ash-based bias/cut-point initialisation and
grid pruning.

Categorical covariates are supported natively via per-column ``nn.Embedding``
tables.  Identifiability of the softmax embedding is fixed by a hard
reference-category gauge (row 0 of every embedding pinned at zero), enforced
by a backward hook installed via ``_install_reference_gauge``.
"""

import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cebmf_torch.cebnm.cash_solver import (
    cash_PosteriorMeanNorm,
    pen_loglik_loss,
)
from cebmf_torch.ebnm.ash import PriorType, ash
from cebmf_torch.utils.distribution_operation import get_data_loglik_normal_torch
from cebmf_torch.utils.mixture import autoselect_scales_mix_norm

# ============================================================
# Model classes
# ============================================================


class LcashNet(nn.Module):
    """Multinomial logistic regression: features -> mixture weights.

    Continuous features (if any) feed an ``nn.Linear(cont_dim, num_classes,
    bias=False)``; categorical columns each feed an ``nn.Embedding(T_d,
    num_classes)``.  A separate learnable bias completes the logits, which
    are softmaxed to give per-observation mixture weights.

    The reference-category gauge (row 0 of every embedding pinned at zero)
    is enforced externally by :func:`_install_reference_gauge`.

    Parameters
    ----------
    cont_dim : int
        Number of continuous input features.  May be 0 (categorical-only).
    num_classes : int
        Number of mixture components (output classes).
    cat_n_levels : list[int] | None
        Per-column number of levels for categorical inputs.  ``None`` or an
        empty list means no categorical head.
    log_pi_init : torch.Tensor or None
        If provided, (K,) tensor of centred log-weights from a global ash
        fit.  Used to initialise the bias so that ``softmax(bias)``
        approximates the global ash pi when all feature coefficients are
        zero.
    generator : torch.Generator or None
        Optional generator for reproducible weight initialisation of the
        continuous head.
    """

    def __init__(
        self,
        cont_dim: int,
        num_classes: int,
        cat_n_levels: list[int] | None = None,
        log_pi_init: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        if cont_dim > 0:
            self.cont = nn.Linear(cont_dim, num_classes, bias=False)
            # Small random perturbation breaks symmetry across features.
            # Starting from exact zeros leads Adam to different local
            # optima on high-dimensional feature sets (F > 100).
            nn.init.normal_(self.cont.weight, mean=0.0, std=0.01, generator=generator)
        else:
            self.cont = None

        self.cat = nn.ModuleList([nn.Embedding(t, num_classes) for t in (cat_n_levels or [])])
        for emb in self.cat:
            nn.init.zeros_(emb.weight)

        self.bias = nn.Parameter(torch.zeros(num_classes))
        if log_pi_init is not None:
            with torch.no_grad():
                self.bias.copy_(log_pi_init)

    def forward(
        self,
        x_cont: torch.Tensor | None,
        x_cat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute mixture weights pi_k for each observation.

        Either ``x_cont`` or ``x_cat`` may be ``None`` (but not both).
        """
        if x_cont is None and x_cat is None:
            raise ValueError("LcashNet.forward requires at least one of x_cont, x_cat.")

        n = x_cont.shape[0] if x_cont is not None else x_cat.shape[0]
        logits = self.bias.expand(n, -1).clone()
        if self.cont is not None and x_cont is not None:
            logits = logits + self.cont(x_cont)
        if x_cat is not None:
            for d, emb in enumerate(self.cat):
                logits = logits + emb(x_cat[:, d])
        return torch.softmax(logits, dim=1)


class PropOddsLcashNet(nn.Module):
    """Proportional odds (ordered logistic) mapping: features -> mixture weights.

    A shared weight vector maps continuous features to a scalar signal
    strength, and per-column scalar embeddings ``nn.Embedding(T_d, 1)`` add
    categorical contributions to the same scalar score.  K-1 ordered
    cut-points convert the score to mixture weights via cumulative
    logistic probabilities.

    Parameters
    ----------
    cont_dim : int
        Number of continuous input features.  May be 0 (categorical-only).
    num_classes : int
        Number of mixture components (K).
    cat_n_levels : list[int] | None
        Per-column number of levels for categorical inputs.
    log_pi_init : torch.Tensor or None
        If provided, (K,) tensor of centred log-weights from a global ash
        fit.  Used to initialise ordered cut-points so that the model
        recovers the global ash pi when all feature coefficients are zero.
    generator : torch.Generator or None
        Optional generator for reproducible weight initialisation of the
        continuous head.
    """

    def __init__(
        self,
        cont_dim: int,
        num_classes: int,
        cat_n_levels: list[int] | None = None,
        log_pi_init: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        K = num_classes

        # Shared continuous weights (initialised near zero so the model
        # starts close to the exchangeable prior).  When ``cont_dim == 0``
        # we still register an empty parameter so the optimiser parameter
        # group construction is uniform; it carries no learnable degrees
        # of freedom and is omitted from the optimiser if empty.
        if cont_dim > 0:
            self.w = nn.Parameter(torch.empty(cont_dim))
            nn.init.normal_(self.w, mean=0.0, std=0.01, generator=generator)
        else:
            self.w = None

        # Per-column scalar embeddings.
        self.cat = nn.ModuleList([nn.Embedding(t, 1) for t in (cat_n_levels or [])])
        for emb in self.cat:
            nn.init.zeros_(emb.weight)

        # Cut-point parameterisation: delta_1 (free), delta_2..K-1 (gaps)
        if log_pi_init is not None and K > 1:
            init_cuts = self._init_cutpoints_from_pi(log_pi_init, K)
        else:
            init_cuts = torch.linspace(-2.0, 2.0, K - 1)

        self.delta_1 = nn.Parameter(init_cuts[0:1])  # (1,)
        if K > 2:
            gaps = torch.log(torch.clamp(init_cuts[1:] - init_cuts[:-1], min=1e-6))
            self.delta_gaps = nn.Parameter(gaps)  # (K-2,)
        else:
            self.delta_gaps = None

        self._K = K

    @staticmethod
    def _init_cutpoints_from_pi(log_pi_init: torch.Tensor, K: int) -> torch.Tensor:
        """Initialise cut-points so that sigma(theta_k) approx cumprob_k.

        At initialisation w ~ 0, so s_i ~ 0 for all genes.  Then
        pi_k = sigma(theta_{k+1}) - sigma(theta_k), so we need
        sigma(theta_k) = sum_{j<k} pi_j, i.e. theta_k = logit(cumprob_k).
        """
        pi = torch.exp(log_pi_init - log_pi_init.max())
        pi = pi / pi.sum()
        cumprob = torch.cumsum(pi, dim=0)[:-1]  # K-1 values
        cumprob = torch.clamp(cumprob, 1e-6, 1 - 1e-6)
        cuts = torch.log(cumprob / (1 - cumprob))
        return cuts

    def _get_cutpoints(self) -> torch.Tensor:
        """Reconstruct ordered cut-points from unconstrained parameters."""
        # K=1: degenerate case, all weight on the single component.
        if self._K == 1:
            return torch.empty(0, device=self.delta_1.device)
        if self.delta_gaps is not None:
            gaps = torch.exp(self.delta_gaps)
            return torch.cat([self.delta_1, self.delta_1 + torch.cumsum(gaps, dim=0)])
        return self.delta_1  # K = 2: single cut-point

    def forward(
        self,
        x_cont: torch.Tensor | None,
        x_cat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute mixture weights pi_k for each observation.

        Either ``x_cont`` or ``x_cat`` may be ``None`` (but not both).
        """
        if x_cont is None and x_cat is None:
            raise ValueError("PropOddsLcashNet.forward requires at least one of x_cont, x_cat.")

        n = x_cont.shape[0] if x_cont is not None else x_cat.shape[0]
        device = x_cont.device if x_cont is not None else x_cat.device

        s = torch.zeros(n, device=device)
        if self.w is not None and x_cont is not None:
            s = s + x_cont @ self.w
        if x_cat is not None:
            for d, emb in enumerate(self.cat):
                s = s + emb(x_cat[:, d]).squeeze(-1)

        theta = self._get_cutpoints()  # (K-1,)

        # Cumulative probabilities: P(category <= k) = sigma(theta_k - s)
        cum_probs = torch.sigmoid(theta.unsqueeze(0) - s.unsqueeze(1))  # (G, K-1)

        # Convert cumulative to category probabilities
        ones = torch.ones(n, 1, device=device)
        zeros = torch.zeros(n, 1, device=device)
        cum_ext = torch.cat([zeros, cum_probs, ones], dim=1)  # (G, K+1)
        pi = cum_ext[:, 1:] - cum_ext[:, :-1]  # (G, K)

        # Numerical safety: clamp small negatives from floating-point
        pi = torch.clamp(pi, min=1e-10)
        pi = pi / pi.sum(dim=1, keepdim=True)

        return pi


# ============================================================
# Reference-category gauge
# ============================================================


def _install_reference_gauge(net: nn.Module) -> None:
    """Pin row 0 of every categorical embedding at zero.

    Called once after ``net`` is constructed (and after any warm-start load
    via ``model_param``).  The hook zeros the gradient on row 0 of each
    categorical embedding.  Row 0 is also explicitly zeroed here, defensively,
    in case a warm-start populated row 0 with non-zero values; without this
    the gauge would pin row 0 at the loaded (non-zero) value.

    Works on both :class:`LcashNet` and :class:`PropOddsLcashNet` because
    both expose ``.cat`` as the ``nn.ModuleList`` of categorical embeddings.

    Assumption: the training loop calls ``optimizer.zero_grad()`` between
    backward passes (the standard PyTorch convention).  The hook clones
    ``grad`` to avoid in-place mutation of the autograd graph.
    """
    for emb in net.cat:
        if emb.num_embeddings < 2:
            raise ValueError("Categorical column with fewer than 2 levels is degenerate.")
        with torch.no_grad():
            emb.weight[0].zero_()  # defensive: handle warm-start case

        def _zero_row_zero(grad, _emb=emb):
            grad = grad.clone()
            grad[0].zero_()
            return grad

        emb.weight.register_hook(_zero_row_zero)


# ============================================================
# Shared helpers
# ============================================================


def _validate_and_normalise_cat(
    X_cat: torch.Tensor | None,
    n_cat_levels: int | list[int] | None,
) -> tuple[torch.Tensor | None, list[int] | None]:
    """Validate and normalise categorical inputs.

    Promotes a 1-D ``X_cat`` (and scalar ``n_cat_levels``) to the canonical
    ``(N, F_d)`` / ``[T_1, ..., T_{F_d}]`` shapes, and checks dtype, range
    and per-column level counts.
    """
    if X_cat is None:
        if n_cat_levels is not None:
            raise ValueError("n_cat_levels was provided without X_cat; either supply X_cat or omit n_cat_levels.")
        return None, None

    if not isinstance(X_cat, torch.Tensor):
        X_cat = torch.as_tensor(X_cat)

    if X_cat.dtype != torch.long:
        raise TypeError(
            "X_cat must be a torch.long tensor of category indices "
            f"(got dtype {X_cat.dtype}); use X for continuous covariates."
        )

    if n_cat_levels is None:
        raise ValueError("n_cat_levels is required when X_cat is provided.")

    # Promote 1-D X_cat to (N, 1) and scalar n_cat_levels to length-1 list.
    if X_cat.ndim == 1:
        X_cat = X_cat.reshape(-1, 1)
    if X_cat.ndim != 2:
        raise ValueError(f"X_cat must be 1-D or 2-D (got ndim={X_cat.ndim}).")

    if isinstance(n_cat_levels, int):
        n_cat_levels = [n_cat_levels]
    else:
        n_cat_levels = list(n_cat_levels)

    if len(n_cat_levels) != X_cat.shape[1]:
        raise ValueError(f"n_cat_levels has length {len(n_cat_levels)} but X_cat has {X_cat.shape[1]} columns.")

    for d, t in enumerate(n_cat_levels):
        if t < 2:
            raise ValueError(
                f"Categorical column {d} has n_cat_levels={t}; columns with fewer than 2 levels are degenerate."
            )
        col = X_cat[:, d]
        if col.numel() > 0:
            col_min = int(col.min().item())
            col_max = int(col.max().item())
            if col_min < 0:
                raise ValueError(
                    f"X_cat column {d} contains negative index {col_min}; indices must lie in [0, n_cat_levels[d])."
                )
            if col_max >= t:
                raise ValueError(
                    f"X_cat column {d} contains index {col_max} but "
                    f"n_cat_levels[{d}]={t}; indices must be < n_cat_levels."
                )

    return X_cat, n_cat_levels


def _validate_inputs(
    X: torch.Tensor | None,
    X_cat: torch.Tensor | None,
    n_cat_levels: int | list[int] | None,
) -> tuple[torch.Tensor | None, list[int] | None]:
    """Top-level input validation shared by both entry points.

    Returns the normalised ``(X_cat, n_cat_levels)`` pair.  ``X`` is not
    modified here; standardisation happens in :func:`_prepare_inputs`.
    """
    if X is None and X_cat is None:
        raise ValueError("At least one of X (continuous) or X_cat (categorical) must be provided.")
    return _validate_and_normalise_cat(X_cat, n_cat_levels)


def _prepare_inputs(
    X: torch.Tensor | None,
    X_cat: torch.Tensor | None,
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Convert inputs to tensors on device; standardise X (NaN-aware).

    Categorical inputs bypass standardisation entirely; they are only moved
    to ``device`` and kept as ``torch.long``.

    The continuous standardisation is NaN-aware: mean and std are computed
    on non-NaN values only, then NaN positions are zero-filled.  This
    ensures that missing features contribute nothing to the logits and
    that the statistics are not biased by the zero-fill.
    """
    if X is not None:
        X = torch.as_tensor(X, dtype=torch.float32, device=device)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_scaled = _nanstandardise(X)
    else:
        X_scaled = None

    if X_cat is not None:
        X_cat = X_cat.to(device=device, dtype=torch.long)

    betahat = torch.as_tensor(betahat, dtype=torch.float32, device=device)
    sebetahat = torch.as_tensor(sebetahat, dtype=torch.float32, device=device)
    return X_scaled, X_cat, betahat, sebetahat


def _nanstandardise(X: torch.Tensor) -> torch.Tensor:
    """Standardise columns using non-NaN values, then zero-fill NaN.

    Vectorised implementation. For each column, compute mean and
    population std on observed (non-NaN) entries, standardise observed
    values, and set NaN positions to 0. Columns with zero std
    (constant or all-NaN) are set to 0.
    """
    mask = ~torch.isnan(X)
    counts = mask.sum(dim=0)  # (F,)

    # Replace NaN with 0 for safe summation
    X_filled = torch.where(mask, X, torch.zeros_like(X))

    # Mean on observed values
    safe_counts = counts.clamp(min=1)
    mu = X_filled.sum(dim=0) / safe_counts  # (F,)

    # Population std on observed values
    diff = torch.where(mask, X - mu, torch.zeros_like(X))
    var = (diff**2).sum(dim=0) / safe_counts  # (F,)
    sd = var.sqrt()

    # Standardise observed, zero-fill missing
    safe_sd = torch.where((sd > 0) & (counts > 1), sd, torch.ones_like(sd))
    X_out = torch.where(
        mask & (sd > 0).unsqueeze(0) & (counts > 1).unsqueeze(0),
        diff / safe_sd,
        torch.zeros_like(X),
    )
    return X_out


def _select_grid(
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    mult: float,
    ash_init: bool,
    ash_threshold: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Select mixture grid and (optionally) initialise from ash.

    Always builds the grid via ``autoselect_scales_mix_norm(mult=mult)``.
    When ``ash_init=True``, additionally runs a full ash fit (L-BFGS
    optimizer) to determine which components are active and initialise
    the bias/cut-points from the ash mixture weights.

    Parameters
    ----------
    mult : float
        Multiplicative step between grid SDs.  Smaller values give a
        finer grid with more components (sqrt(2) ≈ 27 components,
        2.0 ≈ 15 components for typical data).
    ash_init : bool
        If True, run ash internally with ``optimizer="lbfgs"`` to
        prune the grid to active components and initialise bias from
        the ash weights.
    ash_threshold : float
        Pruning threshold: components with ``pi <= ash_threshold``
        are dropped.  Only used when ``ash_init=True``.

    Returns
    -------
    scale : tensor (K,)
        Mixture component standard deviations.
    log_pi_init : tensor (K,) or None
        Centred log-weights for bias/cut-point initialisation, or
        None when ``ash_init=False``.
    """
    if ash_init:
        ash_result = ash(betahat, sebetahat, prior=PriorType.NORM, verbose=False, optimizer="lbfgs", mult=mult)
        pi_full = ash_result.pi
        active = pi_full > ash_threshold
        # Fallback: ensure at least K=2 (spike + one slab)
        if active.sum() < 2:
            active = torch.zeros_like(pi_full, dtype=torch.bool)
            active[0] = True
            non_spike = pi_full.clone()
            non_spike[0] = -1.0
            active[non_spike.argmax()] = True
        scale = ash_result.scale[active].to(device=device, dtype=torch.float32)
        pi_active = pi_full[active]
        log_pi_init = torch.log(pi_active.clamp(min=1e-30))
        log_pi_init = log_pi_init - log_pi_init.mean()
        log_pi_init = log_pi_init.to(device=device, dtype=torch.float32)
        return scale, log_pi_init

    scale = autoselect_scales_mix_norm(betahat=betahat, sebetahat=sebetahat, mult=mult)
    if not isinstance(scale, torch.Tensor):
        scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
    else:
        scale = scale.to(device=device, dtype=torch.float32)
    return scale, None


def _train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X_scaled: torch.Tensor | None,
    X_cat: torch.Tensor | None,
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    scale: torch.Tensor,
    n_epochs: int,
    batch_size: int,
    penalty: float,
    verbose: bool,
    label: str,
    seed: int = 42,
) -> float:
    """Run the training loop. Returns the final-epoch total loss.

    Pre-computes the (G, K) log-likelihood matrix once rather than
    recomputing per mini-batch (logL is constant during training).
    Batch ordering is seeded for reproducibility via a ``DataLoader`` over
    a ``TensorDataset`` that keeps continuous, categorical and outcome
    tensors aligned.
    """
    model.train()
    if X_scaled is not None:
        device = X_scaled.device
        n = X_scaled.shape[0]
    else:
        device = X_cat.device
        n = X_cat.shape[0]

    # Pre-compute log-likelihood matrix (constant during training).
    loc = torch.zeros_like(scale)
    with torch.no_grad():
        logL_all = get_data_loglik_normal_torch(
            betahat=betahat,
            sebetahat=sebetahat,
            location=loc,
            scale=scale,
        )

    # We index tensors by integer position rather than passing them
    # through the DataLoader collate to avoid repeated allocation of
    # large minibatches.  The DataLoader exists only as a seeded
    # permutation generator.
    g = torch.Generator()
    g.manual_seed(seed)
    index_ds = TensorDataset(torch.arange(n))
    loader = DataLoader(index_ds, batch_size=batch_size, shuffle=True, generator=g)

    n_batches = max(1, len(loader))

    final_epoch_loss = 0.0
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for (idx,) in loader:
            idx = idx.to(device)
            x_cont_b = X_scaled[idx] if X_scaled is not None else None
            x_cat_b = X_cat[idx] if X_cat is not None else None
            pi_pred = model(x_cont_b, x_cat_b)
            loss = pen_loglik_loss(pi_pred, logL_all[idx], penalty=penalty)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        final_epoch_loss = epoch_loss
        if verbose and (epoch + 1) % 50 == 0:
            print(f"[{label}] Epoch {epoch + 1}/{n_epochs} | Loss: {epoch_loss / n_batches:.4f}")

    return final_epoch_loss


def _compute_posteriors(
    model: nn.Module,
    X_scaled: torch.Tensor | None,
    X_cat: torch.Tensor | None,
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    scale: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Vectorised posterior computation with per-observation pi.

    Assumes location = 0 for all mixture components (spike mean is 0).
    This matches the zero-centred normal mixture prior used by LC-ASH.

    Returns
    -------
    post_mean, post_mean2, post_sd, all_pi_values, marginal_loglik
        ``marginal_loglik`` is the full-data marginal log-likelihood
        ``sum_g logsumexp_k (log pi_g,k + log p(beta_g | 0, sqrt(se_g^2 + scale_k^2)))``,
        i.e. ``log p(y | fitted prior)`` without any spike Dirichlet penalty.
        It is what the cebmf consumer at ``cebmf.py:299``
        (``self.kl_l[k] = (-resL.loss) - nm_ll_L``) requires of the loss
        field on this object.
    """
    model.eval()
    loc = torch.zeros_like(scale)
    with torch.no_grad():
        all_pi_values = model(X_scaled, X_cat)  # (G, K)

        data_loglik = get_data_loglik_normal_torch(
            betahat=betahat, sebetahat=sebetahat, location=loc, scale=scale
        )  # (G, K)

        # Use dtype-appropriate eps to avoid log(0) in float32.
        eps = torch.finfo(all_pi_values.dtype).tiny
        log_pi_all = torch.log(torch.clamp(all_pi_values, min=eps))  # (G, K)
        combined = data_loglik + log_pi_all  # (G, K)
        log_norm = torch.logsumexp(combined, dim=1, keepdim=True)  # (G, 1)
        # log_norm[g] is the per-gene marginal log-likelihood of the fitted
        # mixture; summing gives the full-data marginal log-lik (no penalty).
        marginal_loglik = float(log_norm.sum().item())
        resp = torch.exp(combined - log_norm)  # (G, K) responsibilities

        s2 = sebetahat.pow(2).unsqueeze(1)  # (G, 1)
        t2 = scale.pow(2).unsqueeze(0)  # (1, K)

        denom = (1.0 / s2) + torch.where(t2 > 0, 1.0 / t2, torch.zeros_like(t2))
        post_var_comp = torch.where(t2 > 0, 1.0 / denom, torch.zeros_like(denom))  # (G, K)

        m_comp = torch.where(
            t2 > 0,
            post_var_comp * (betahat.unsqueeze(1) / s2),
            torch.zeros(1, device=device),
        )  # (G, K)

        post_mean = torch.sum(resp * m_comp, dim=1)
        post_mean2 = torch.sum(resp * (post_var_comp + m_comp.pow(2)), dim=1)
        post_sd = torch.sqrt(torch.clamp(post_mean2 - post_mean.pow(2), min=0.0))

    return post_mean, post_mean2, post_sd, all_pi_values, marginal_loglik


def _warm_start(
    model: nn.Module,
    model_param: dict | None,
    label: str,
) -> None:
    """Load state dict with a guard against architecture mismatch."""
    if model_param is not None:
        try:
            model.load_state_dict(model_param)
        except RuntimeError:
            warnings.warn(
                f"{label} warm-start skipped: grid size changed between iterations",
                stacklevel=3,
            )


def _build_optimizer(
    model: nn.Module,
    model_class: type,
    weight_decay: float,
    lr: float,
) -> torch.optim.Optimizer:
    """Build Adam with weight_decay applied only to feature weights.

    Embedding tables and bias/cut-points are excluded from weight decay
    (cf. existing `linear.bias` / cut-point exemption).
    """
    if model_class is LcashNet:
        feature_params = []
        if model.cont is not None:
            feature_params.append(model.cont.weight)
        no_decay_params = [model.bias]
        for emb in model.cat:
            no_decay_params.append(emb.weight)
        param_groups = []
        if feature_params:
            param_groups.append({"params": feature_params, "weight_decay": weight_decay})
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    else:  # PropOddsLcashNet
        feature_params = []
        if model.w is not None:
            feature_params.append(model.w)
        cutpoint_params = [model.delta_1]
        if model.delta_gaps is not None:
            cutpoint_params.append(model.delta_gaps)
        no_decay_params = list(cutpoint_params)
        for emb in model.cat:
            no_decay_params.append(emb.weight)
        param_groups = []
        if feature_params:
            param_groups.append({"params": feature_params, "weight_decay": weight_decay})
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return torch.optim.Adam(param_groups, lr=lr)


def _fit_lcash(
    X: torch.Tensor | None,
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    model_class: type,
    label: str,
    n_epochs: int = 200,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    penalty: float = 1.5,
    mult: float = 1.4142135623730951,
    ash_init: bool = True,
    ash_threshold: float = 1e-6,
    model_param: dict | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
    seed: int = 42,
    X_cat: torch.Tensor | None = None,
    n_cat_levels: int | list[int] | None = None,
) -> cash_PosteriorMeanNorm:
    """Shared implementation for both softmax and proportional odds LC-ASH.

    Parameters
    ----------
    model_class : type
        Either ``LcashNet`` or ``PropOddsLcashNet``.
    label : str
        Label for verbose logging (e.g. "LC-ASH" or "PO-LC-ASH").

    See ``lcash_posterior_means`` for other parameter descriptions.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if n_epochs is None:
        n_epochs = 200

    X_cat, cat_levels = _validate_inputs(X, X_cat, n_cat_levels)
    X_scaled, X_cat, betahat, sebetahat = _prepare_inputs(X, X_cat, betahat, sebetahat, device)
    scale, log_pi_init = _select_grid(betahat, sebetahat, mult, ash_init, ash_threshold, device)

    # Local RNG for reproducible weight init and batch ordering.
    # Does not mutate global torch RNG state.
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    K = scale.shape[0]
    cont_dim = X_scaled.shape[1] if X_scaled is not None else 0
    model = model_class(
        cont_dim,
        K,
        cat_n_levels=cat_levels,
        log_pi_init=log_pi_init,
        generator=rng,
    ).to(device)
    _warm_start(model, model_param, label)
    if cat_levels:
        _install_reference_gauge(model)

    optimizer = _build_optimizer(model, model_class, weight_decay=weight_decay, lr=lr)

    _train_model(
        model,
        optimizer,
        X_scaled,
        X_cat,
        betahat,
        sebetahat,
        scale,
        n_epochs,
        batch_size,
        penalty,
        verbose,
        label,
        seed=seed,
    )

    post_mean, post_mean2, post_sd, all_pi_values, marginal_loglik = _compute_posteriors(
        model,
        X_scaled,
        X_cat,
        betahat,
        sebetahat,
        scale,
        device,
    )

    # `loss` is the negative full-data marginal log-likelihood under the
    # fitted prior, *without* the spike Dirichlet penalty. This matches
    # the convention used by `cebnm/emdn.py` and is the meaning required
    # by `cebmf.py`'s per-factor `kl_l[k] = (-loss) - nm_ll_L` formula.
    return cash_PosteriorMeanNorm(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        pi_np=all_pi_values,
        loss=-marginal_loglik,
        scale=scale,
        model_param=model.state_dict(),
    )


# ============================================================
# Public entry points
# ============================================================


def lcash_posterior_means(
    X: torch.Tensor | None,
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    n_epochs: int | None = 200,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    penalty: float = 1.5,
    mult: float = 1.4142135623730951,
    ash_init: bool = True,
    ash_threshold: float = 1e-6,
    model_param: dict | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
    seed: int = 42,
    *,
    X_cat: torch.Tensor | None = None,
    n_cat_levels: int | list[int] | None = None,
) -> cash_PosteriorMeanNorm:
    """LC-ASH: linear covariate-modulated mixture weights.

    Parameters
    ----------
    X : tensor (G, F) or None
        Continuous feature matrix.  Standardised internally with NaN-aware
        statistics (mean/std computed on non-NaN values, NaN positions
        zero-filled).  Pre-standardisation is not required.  May be ``None``
        if ``X_cat`` is provided.
    betahat : tensor (G,)
        Effect estimates.
    sebetahat : tensor (G,)
        Standard errors.
    n_epochs : int or None
        Training epochs.  Inside cEBMF, overridden by ``internal_epoch``.
    batch_size : int
        Mini-batch size for Adam.
    lr : float
        Learning rate.
    weight_decay : float
        L2 penalty on continuous-feature coefficients only (not bias, not
        embedding tables).
    penalty : float
        Dirichlet spike penalty (lambda_pen).  1.0 = no penalty.
    mult : float
        Multiplicative step between mixture grid SDs.  Smaller values
        give a finer grid with more components.  Default sqrt(2) matches
        R ashr and gives ~27 components before pruning.
    ash_init : bool
        If True (default), run ash internally (L-BFGS optimizer) to prune
        the grid to active components and initialise the bias from the
        ash weights, so the model starts at the exchangeable ash solution
        when all feature coefficients are zero.  If False, use the full
        grid with uniform bias initialisation.
    ash_threshold : float
        Pruning threshold: components with ``pi <= threshold`` are dropped.
        Only used when ``ash_init=True``.
    model_param : dict or None
        State dict from a previous call, for warm-starting.
    device : torch.device or None
        Compute device.  Defaults to CUDA if available.
    verbose : bool
        If True (default), print training progress every 50 epochs.
    seed : int
        Random seed for weight initialisation and batch ordering.
    X_cat : tensor (G,) or (G, F_d), torch.long, keyword-only
        Categorical covariate indices.  Each column is treated as an
        independent factor with its own embedding table.  Bypasses
        :func:`_nanstandardise`.  Indices in column ``d`` must satisfy
        ``0 <= idx < n_cat_levels[d]``.  Required ``dtype=torch.long``.
    n_cat_levels : int, list[int] or None, keyword-only
        Per-column number of levels.  Required when ``X_cat`` is given.
        A scalar is promoted to a length-1 list (after also promoting a
        1-D ``X_cat`` to ``(N, 1)``).

    Returns
    -------
    cash_PosteriorMeanNorm
        Container with post_mean, post_mean2, post_sd, pi_np (G, K),
        scale (K,), loss, model_param (state dict for warm-starting).
    """
    return _fit_lcash(
        X,
        betahat,
        sebetahat,
        model_class=LcashNet,
        label="LC-ASH",
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        penalty=penalty,
        mult=mult,
        ash_init=ash_init,
        ash_threshold=ash_threshold,
        model_param=model_param,
        device=device,
        verbose=verbose,
        seed=seed,
        X_cat=X_cat,
        n_cat_levels=n_cat_levels,
    )


def po_lcash_posterior_means(
    X: torch.Tensor | None,
    betahat: torch.Tensor,
    sebetahat: torch.Tensor,
    n_epochs: int | None = 200,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    penalty: float = 1.5,
    mult: float = 1.4142135623730951,
    ash_init: bool = True,
    ash_threshold: float = 1e-6,
    model_param: dict | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
    seed: int = 42,
    *,
    X_cat: torch.Tensor | None = None,
    n_cat_levels: int | list[int] | None = None,
) -> cash_PosteriorMeanNorm:
    """Proportional odds LC-ASH: ordered logistic covariate-modulated weights.

    A shared weight vector maps continuous features to a scalar signal
    strength ``s_i = x_i^T w``.  Per-column scalar embeddings add
    categorical contributions to the same score.  K-1 ordered cut-points
    convert the score to mixture weights via cumulative logistic
    probabilities.  This has F + sum_d T_d + K - 1 parameters
    (vs K * (F + sum_d T_d) for softmax LC-ASH), making it more
    parsimonious when K is large relative to the feature count.

    When ``ash_init=True``, the grid is pruned to ash's active components
    and the cut-points are initialised from the ash weights, so the model
    starts at the exchangeable ash solution.

    Parameters
    ----------
    X : tensor (G, F) or None
        Continuous feature matrix.  Standardised internally with NaN-aware
        statistics.  May be ``None`` if ``X_cat`` is provided.
    betahat, sebetahat, n_epochs, batch_size, lr, weight_decay, penalty,
    mult, ash_init, ash_threshold, model_param, device, verbose, seed :
        See :func:`lcash_posterior_means`.
    X_cat, n_cat_levels :
        See :func:`lcash_posterior_means`.

    Returns
    -------
    cash_PosteriorMeanNorm
        Container with post_mean, post_mean2, post_sd, pi_np (G, K),
        scale (K,), loss, model_param (state dict for warm-starting).
    """
    return _fit_lcash(
        X,
        betahat,
        sebetahat,
        model_class=PropOddsLcashNet,
        label="PO-LC-ASH",
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        penalty=penalty,
        mult=mult,
        ash_init=ash_init,
        ash_threshold=ash_threshold,
        model_param=model_param,
        device=device,
        verbose=verbose,
        seed=seed,
        X_cat=X_cat,
        n_cat_levels=n_cat_levels,
    )
