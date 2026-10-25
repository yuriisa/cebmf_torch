"""Magnitude and direction adaptive shrinkage with half-normal components.

Input columns are explicit finite terms prepared by the caller. The solver
standardizes those columns but does not impute or transform their meaning.
"""

import math
import operator
import warnings
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from cebmf_torch.cebnm.cash_solver import cash_PosteriorMeanNorm
from cebmf_torch.cebnm.lcash import (
    PropOddsLcashNet,
    _as_float64,
    _conditional_loss,
    _feature_statistics,
    _nanstandardise,
    _require_finite,
    _train_model,
    _validate_grid,
)
from cebmf_torch.utils.halfnormal import (
    get_data_loglik_halfnormal_torch,
    positive_normal_moments,
    truncated_normal_log_predictive,
)
from cebmf_torch.utils.maths import _logpdf_normal
from cebmf_torch.utils.mixture import autoselect_scales_mix_norm, mixture_log_penalty, optimize_pi_logL_lbfgs

_ROLES = frozenset(("magnitude", "direction", "both"))
_DEFAULTS = {"penalty": 10.0, "coefficient_prior_scale": 1.0, "coefficient_prior_df": 5.0}
_INITIAL_UNIFORM_MASS = 0.01
_CPU = torch.device("cpu")


@dataclass
class MdAshPrediction:
    """Full latent mixture and its summaries, in float64 on CPU.

    Continuous components are N(component_mean, component_sd**2), conditioned
    on component_sign * theta > 0. The first component is the spike at zero.
    These means and SDs are BEFORE truncation. posterior_weights contains the
    updated weights, or prior weights when observed is false. log_pi always
    contains prior log weights. No density is approximated by these summaries.
    """

    post_mean: torch.Tensor
    post_mean2: torch.Tensor
    post_sd: torch.Tensor
    log_pi: torch.Tensor
    scale: torch.Tensor
    observed: torch.Tensor
    posterior_weights: torch.Tensor
    component_mean: torch.Tensor
    component_sd: torch.Tensor
    component_sign: torch.Tensor
    log_predictive: torch.Tensor | None = None

    @property
    def prob_zero(self):
        return self.posterior_weights[:, 0]

    @property
    def prob_negative(self):
        return self.posterior_weights[:, 1::2].sum(1)

    @property
    def prob_positive(self):
        return self.posterior_weights[:, 2::2].sum(1)


def _tensor(value):
    return _as_float64(value, _CPU).detach()


def _features(X, n):
    X = torch.empty((n, 0), dtype=torch.float64) if X is None else _tensor(X)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or X.shape[0] != n or n == 0:
        raise ValueError("X must be a nonempty matrix with one row per observation.")
    if not torch.isfinite(X).all():
        raise ValueError(
            "X must contain finite prepared terms. Impute missing covariates before fitting or prediction."
        )
    return X


def _roles(value, p, saved=None):
    value = (saved if saved is not None else "magnitude") if value is None else value
    if isinstance(value, str):
        if value not in _ROLES:
            raise ValueError("covariate_roles must contain magnitude, direction, or both.")
        value = (value,) * p
    elif isinstance(value, Sequence):
        value = tuple(value)
    else:
        raise TypeError("covariate_roles must be a role or one role per X column.")
    if len(value) != p or any(not isinstance(v, str) or v not in _ROLES for v in value):
        raise ValueError("Supply one valid covariate role per X column.")
    if saved is not None and value != tuple(saved):
        raise ValueError("covariate_roles differs from the saved fit. Refit to change roles.")
    return value


def _active_columns(roles, sd):
    return tuple(
        tuple(i for i, r in enumerate(roles) if sd[i] > 0 and r in (channel, "both"))
        for channel in ("magnitude", "direction")
    )


class MdAshNet(nn.Module):
    """One slope vector per channel, with component or shared sign intercepts."""

    def __init__(self, magnitude_columns, direction_columns, log_pi_init, sign_intercept="shared"):
        super().__init__()
        self.magnitude_columns, self.direction_columns = magnitude_columns, direction_columns
        pairs = log_pi_init[1:].reshape(-1, 2)
        log_q = torch.cat((log_pi_init[:1], torch.logsumexp(pairs, dim=1)))
        self.magnitude = PropOddsLcashNet(len(magnitude_columns), log_q.numel(), log_pi_init=log_q)
        mode = _sign_intercept(sign_intercept)
        intercepts = pairs[:, 1] - pairs[:, 0]
        if mode == "shared":
            intercepts = (torch.logsumexp(pairs[:, 1], 0) - torch.logsumexp(pairs[:, 0], 0)).reshape(1)
        self.sign_intercepts = nn.Parameter(intercepts.clone())
        self.direction_weight = nn.Parameter(log_pi_init.new_zeros(len(direction_columns)))

    @property
    def w(self):
        """All active slopes for the unchanged upstream Student-t objective."""
        return torch.cat((self.magnitude.w, self.direction_weight))

    def log_probs(self, X):
        log_q = self.magnitude.log_probs(X[:, self.magnitude_columns])
        direction = X[:, self.direction_columns] @ self.direction_weight
        logits = self.sign_intercepts[None, :] + direction[:, None]
        signed = torch.stack((F.logsigmoid(-logits), F.logsigmoid(logits)), dim=2)
        return torch.cat((log_q[:, :1], (log_q[:, 1:, None] + signed).flatten(1)), dim=1)


def _sign_intercept(value=None, state=None):
    saved = "shared" if state is None else state.get("sign_intercept", "component")
    if saved not in ("component", "shared"):
        raise ValueError("Saved sign_intercept must be component or shared.")
    value = saved if value is None else value
    if value not in ("component", "shared"):
        raise ValueError("sign_intercept must be component or shared.")
    if state is not None and value != saved:
        raise ValueError("sign_intercept differs from the saved fit. Refit to change it.")
    return value


def _fit_kind(columns, scale, sign_intercept):
    # Shared intercepts constrain even a fit without covariates.
    return "conditional" if len(scale) > 1 and (any(columns) or sign_intercept == "shared") else "exchangeable"


def _options(penalty, coefficient_prior_scale, coefficient_prior_df, state):
    saved = _DEFAULTS if state is None else state["fit_options"]
    options = {
        "penalty": saved["penalty"] if penalty is None else penalty,
        "coefficient_prior_scale": saved["coefficient_prior_scale"]
        if coefficient_prior_scale == "inherit"
        else coefficient_prior_scale,
        "coefficient_prior_df": saved["coefficient_prior_df"]
        if coefficient_prior_df is None
        else coefficient_prior_df,
    }
    if not math.isfinite(options["penalty"]) or options["penalty"] < 1:
        raise ValueError("penalty must be finite and at least one.")
    for name in ("coefficient_prior_scale", "coefficient_prior_df"):
        value = options[name]
        if name == "coefficient_prior_scale" and value is None:
            continue
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    return options


def _restore_state(state, X, roles=None, scale=None):
    required = {
        "version",
        "solver",
        "prior",
        "kind",
        "scale",
        "covariate_roles",
        "feature_mean",
        "feature_sd",
        "fit_options",
        "optimization",
    }
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError("model_param must be complete MD-ASH fitted state.")
    if state["version"] != 1 or state["solver"] != "MD-ASH" or state["prior"] != "halfnormal":
        raise ValueError("Incompatible MD-ASH state. Expected half-normal version 1 state.")
    if not isinstance(state["fit_options"], dict) or set(state["fit_options"]) != set(_DEFAULTS):
        raise ValueError("Saved fit_options must contain the three penalty settings.")
    _options(None, "inherit", None, state)
    roles = _roles(roles, X.shape[1], state["covariate_roles"])
    grid, mean, sd = (_tensor(state[k]).clone() for k in ("scale", "feature_mean", "feature_sd"))
    _validate_grid(grid)
    if scale is not None and not torch.equal(_tensor(scale), grid):
        raise ValueError("scale differs from the saved fit. Refit to change the grid.")
    if mean.shape != (X.shape[1],) or sd.shape != mean.shape:
        raise ValueError("Saved feature statistics do not match X columns.")
    if not torch.isfinite(mean).all() or not torch.isfinite(sd).all() or (sd < 0).any():
        raise ValueError("Saved feature means and SDs must be finite, with nonnegative SDs.")
    columns = _active_columns(roles, sd)
    kind = _fit_kind(columns, grid, _sign_intercept(state=state))
    if state["kind"] != kind:
        raise ValueError("Saved fit kind does not agree with its features and grid.")
    return grid, mean, sd, roles


def _exchangeable_weights(state, n_components):
    if "pi" not in state:
        raise ValueError("Exchangeable state requires pi.")
    pi = _tensor(state["pi"]).clone()
    if (
        pi.shape != (n_components,)
        or not torch.isfinite(pi).all()
        or (pi < 0).any()
        or not torch.isclose(pi.sum(), pi.new_tensor(1.0), rtol=0, atol=1e-10)
    ):
        raise ValueError("Saved pi must be finite normalized nonnegative component weights.")
    return pi


def _conditional_model(columns, n_components, state=None, pi=None, sign_intercept=None):
    # Interior starting weights prevent saturated sign logits from suppressing
    # useful slope gradients. This is initialization, not a fitted weight floor.
    pi = torch.full((n_components,), 1 / n_components, dtype=torch.float64) if pi is None else pi
    initial = ((1 - _INITIAL_UNIFORM_MASS) * pi + _INITIAL_UNIFORM_MASS / n_components).log()
    initial -= torch.logsumexp(initial, 0)
    model = MdAshNet(*columns, initial, sign_intercept=_sign_intercept(sign_intercept, state))
    if state is not None:
        if "state_dict" not in state:
            raise ValueError("Conditional state requires state_dict.")
        model.load_state_dict({k: _tensor(v).clone() for k, v in state["state_dict"].items()})
        for value in model.parameters():
            _require_finite(value, "Saved MD-ASH coefficient")
    return model


@torch.no_grad()
def _continuation_start(model, X, likelihood, penalty):
    """Compare saved, interior, and reflected starts before repeated updates.

    Reflection reverses the prior's signs. Interior starts reuse the fresh-fit
    weight mixture at zero standardized covariates and preserve the slopes.
    The Student-t penalty is identical across these candidates. Select using
    the native objective without that common term. No extra fit is required.
    """
    pi = model.log_probs(X.new_zeros((1, X.shape[1])))[0].exp()
    mode = "shared" if model.sign_intercepts.numel() == 1 else "component"
    interior = _conditional_model(
        (model.magnitude_columns, model.direction_columns),
        len(pi),
        pi=pi,
        sign_intercept=mode,
    )
    interior.magnitude.w.copy_(model.magnitude.w)
    interior.direction_weight.copy_(model.direction_weight)
    candidates = [("saved", model), ("interior", interior)]
    for label, original in list(candidates):
        reflected = deepcopy(original)
        reflected.sign_intercepts.neg_()
        reflected.direction_weight.neg_()
        candidates.append(("reflected_" + label, reflected))
    return min(candidates, key=lambda pair: _conditional_loss(pair[1], X, likelihood, penalty, None).item())


@torch.no_grad()
def _exchangeable_start(pi, likelihood, penalty):
    """Apply the same interior and sign-reflection choices to mixture weights."""
    interior = (1 - _INITIAL_UNIFORM_MASS) * pi + _INITIAL_UNIFORM_MASS / len(pi)
    candidates = [("saved", pi), ("interior", interior)]
    for label, original in list(candidates):
        reflected = torch.cat((original[:1], original[1:].reshape(-1, 2).flip(1).flatten()))
        candidates.append(("reflected_" + label, reflected))
    return min(
        candidates,
        key=lambda pair: (
            -torch.logsumexp(likelihood + pair[1].log(), 1).sum() - mixture_log_penalty(pair[1], penalty)
        ).item(),
    )


def _prior_weights(X, state):
    scale, mean, sd, roles = _restore_state(state, X)
    if state["kind"] == "exchangeable":
        log_pi = _exchangeable_weights(state, 2 * len(scale) - 1).log().expand(len(X), -1)
    else:
        model = _conditional_model(_active_columns(roles, sd), 2 * len(scale) - 1, state)
        log_pi = model.log_probs(_nanstandardise(X, mean, sd))
    if torch.isnan(log_pi).any() or torch.isposinf(log_pi).any():
        raise FloatingPointError("Invalid MD-ASH prior log weights.")
    if not torch.allclose(torch.logsumexp(log_pi, 1), log_pi.new_zeros(len(X)), atol=1e-10, rtol=0):
        raise ValueError("Prior weights must sum to one.")
    return log_pi, scale


@torch.no_grad()
def predict_md_ash(X, betahat=None, sebetahat=None, *, model_param, observed=None, target=None, target_se=None):
    """Predict without fitting, using finite prepared input columns.

    Omit source arrays for prior-only prediction. Otherwise NaN source values
    imply unobserved rows, unless an explicit boolean observed mask is supplied.
    Only observed source rows require finite effects and positive SEs. Supply a
    target-eligible subset and both target arrays for analytic predictive LPD.
    Missing covariates must be prepared externally using saved source statistics.
    """
    if (betahat is None) != (sebetahat is None):
        raise ValueError("Supply both betahat and sebetahat, or neither.")
    if (target is None) != (target_se is None):
        raise ValueError("Supply both target and target_se, or neither.")
    reference = next((v for v in (X, betahat, target, observed) if v is not None), None)
    n = 1 if reference is None else len(reference)
    X = _features(X, n)
    log_pi, scale = _prior_weights(X, model_param)

    def vector(value, name):
        value = _tensor(value)
        if value.shape != (n,):
            raise ValueError(f"{name} must have one value per prediction row.")
        return value

    y = se = None
    if betahat is not None:
        y, se = vector(betahat, "betahat"), vector(sebetahat, "sebetahat")
    if observed is None:
        observed = torch.zeros(n, dtype=torch.bool) if y is None else ~(torch.isnan(y) | torch.isnan(se))
    else:
        observed = torch.as_tensor(observed, device="cpu")
        if observed.dtype != torch.bool or observed.shape != (n,):
            raise ValueError("observed must be a boolean vector with one entry per row.")
    if observed.any() and (
        y is None
        or not torch.isfinite(y[observed]).all()
        or not torch.isfinite(se[observed]).all()
        or (se[observed] <= 0).any()
    ):
        raise ValueError("Observed source rows require finite effects and finite positive standard errors.")

    component_sign = torch.cat((scale.new_zeros(1), scale.new_tensor([-1, 1]).repeat(len(scale) - 1)))
    component_mean = log_pi.new_zeros(log_pi.shape)
    component_sd = torch.cat((scale[:1], scale[1:].repeat_interleave(2))).expand(n, -1).clone()
    weights = log_pi.exp()
    log_weights = log_pi.clone()
    if observed.any():
        yy, ss = y[observed], se[observed]
        likelihood = get_data_loglik_halfnormal_torch(yy, ss, scale)
        weights[observed] = (log_pi[observed] + likelihood).softmax(1)
        log_weights[observed] = (log_pi[observed] + likelihood).log_softmax(1)
        ratio = scale[1:] / torch.hypot(ss[:, None], scale[1:])
        component_mean[observed, 1:] = (ratio.square() * yy[:, None]).repeat_interleave(2, 1)
        component_sd[observed, 1:] = (ss[:, None] * ratio).repeat_interleave(2, 1)
    signed_mean, second, variance = positive_normal_moments(
        component_sign[1:] * component_mean[:, 1:] / component_sd[:, 1:],
        component_sd[:, 1:],
        return_variance=True,
    )
    component_actual_mean = signed_mean * component_sign[1:]
    post_mean = (weights[:, 1:] * component_actual_mean).sum(1)
    post_mean2 = torch.maximum((weights[:, 1:] * second).sum(1), post_mean.square())
    post_var = (weights[:, 1:] * (variance + (component_actual_mean - post_mean[:, None]).square())).sum(1)
    post_sd = (post_var + weights[:, 0] * post_mean.square()).sqrt()
    log_predictive = None
    if target is not None:
        z, t = vector(target, "target"), vector(target_se, "target_se")
        if not torch.isfinite(z).all() or not torch.isfinite(t).all() or (t <= 0).any():
            raise ValueError("Targets require finite effects and finite positive standard errors.")

        spike = _logpdf_normal(z, torch.zeros_like(z), t)[:, None]
        continuous = truncated_normal_log_predictive(
            z[:, None], t[:, None], component_mean[:, 1:], component_sd[:, 1:], component_sign[1:]
        )
        log_predictive = torch.logsumexp(log_weights + torch.cat((spike, continuous), 1), 1)
        _require_finite(log_predictive, "MD-ASH predictive log density")
    for name, value in (("mean", post_mean), ("second moment", post_mean2), ("SD", post_sd)):
        _require_finite(value, f"MD-ASH posterior {name}")
    return MdAshPrediction(
        post_mean,
        post_mean2,
        post_sd,
        log_pi,
        scale,
        observed.clone(),
        weights,
        component_mean,
        component_sd,
        component_sign,
        log_predictive,
    )


def md_ash_posterior_means(
    X,
    betahat,
    sebetahat,
    *,
    covariate_roles=None,
    sign_intercept=None,
    scale=None,
    max_iter=None,
    penalty=None,
    coefficient_prior_scale: float | None | Literal["inherit"] = "inherit",
    coefficient_prior_df=None,
    mult=2**0.5,
    model_param=None,
    device=None,
    verbose=False,
):
    """Fit half-normal MD-ASH to eligible source rows only.

    X contains finite completed terms. Roles default to magnitude. The grid is
    [0, a1, ..., aK], with a the Gaussian SD before truncation. Output component
    order is spike, negative_a1, positive_a1, and so on. Outputs are CPU float64.
    Fresh defaults are penalty 10 and Student-t slope scale 1, df 5. Omitted
    settings inherit saved penalties. Explicit coefficient_prior_scale=None
    disables the slope penalty. max_iter defaults to 500 conditional or 2000
    exchangeable iterations. Conditional initialization has a separate 2000
    iteration budget. Saved max_iter=0 evaluates without fitting; a fresh zero
    iteration call is rejected. Fresh conditional fits start from the exchangeable
    weights mixed with 1% uniform mass. The fitted weights have no floor.
    sign_intercept is "shared" (fresh default) or "component". Shared fits use
    one unpenalized sign intercept across widths, including without covariates.
    None inherits the saved mode on continuation. Old states imply component.
    Shared fits use the conditional optimizer and its iteration budget even
    without covariates. A spike-only grid has no sign intercept to estimate.
    Fitting and prediction use CPU float64. Set device="cpu" in cEBMF.
    Continuation uses saved coefficients, grid, and covariate statistics.
    Positive-iteration continuation compares saved, interior, and sign-reflected
    starting values on the current data. It fits only the best starting value.
    Fresh fits and max_iter=0 replay do not use this continuation safeguard.
    The endpoint diagnostics do not establish a global optimum.
    """
    requested_device = torch.device(device if device is not None else getattr(betahat, "device", _CPU))
    if requested_device.type != "cpu":
        raise ValueError("MD-ASH currently supports CPU fitting. Set device to cpu.")
    y, se = _tensor(betahat), _tensor(sebetahat)
    if y.ndim != 1 or se.shape != y.shape or not y.numel():
        raise ValueError("betahat and sebetahat must be nonempty matching vectors.")
    if not torch.isfinite(y).all() or not torch.isfinite(se).all() or (se <= 0).any():
        raise ValueError("Effects must be finite with finite positive standard errors.")
    X = _features(X, len(y))
    if not math.isfinite(mult) or mult <= 1:
        raise ValueError("mult must be finite and greater than one.")
    if max_iter is not None:
        try:
            max_iter = operator.index(max_iter)
        except TypeError:
            raise ValueError("max_iter must be a nonnegative integer or None.") from None
        if max_iter < 0 or (max_iter == 0 and model_param is None):
            raise ValueError("max_iter must be positive for a fresh fit; zero requires model_param.")
    if model_param is None:
        roles = _roles(covariate_roles, X.shape[1])
        mean, sd = _feature_statistics(X)
        scale = autoselect_scales_mix_norm(y, se, mult=mult) if scale is None else _tensor(scale)
    else:
        scale, mean, sd, roles = _restore_state(model_param, X, covariate_roles, scale)
    _validate_grid(scale)
    options = _options(penalty, coefficient_prior_scale, coefficient_prior_df, model_param)
    columns = _active_columns(roles, sd)
    sign_intercept = _sign_intercept(sign_intercept, model_param)
    kind = _fit_kind(columns, scale, sign_intercept)
    budget = (500 if kind == "conditional" else 2000) if max_iter is None else max_iter
    likelihood = get_data_loglik_halfnormal_torch(y, se, scale)
    optimization = None
    if model_param is not None and budget == 0:
        previous = model_param["optimization"]
        optimization = {
            "converged": None,
            "status": "not_assessed",
            "reason": "Saved parameters evaluated without optimization on these inputs.",
            "max_iter": 0,
            "n_iter": 0,
            "func_evals": 0,
            "previous_fit": deepcopy(previous.get("previous_fit", previous)),
        }
    initialization = None
    pi = None
    continuation_start = None
    if model_param is None or (kind == "exchangeable" and budget > 0):
        start = None if model_param is None else _exchangeable_weights(model_param, likelihood.shape[1])
        if start is not None:
            continuation_start, start = _exchangeable_start(start, likelihood, options["penalty"])
        pi, initialization = optimize_pi_logL_lbfgs(
            likelihood,
            options["penalty"],
            zero_threshold=0,
            pi_init=start,
            max_iter=budget if kind == "exchangeable" else 2000,
            return_diagnostics=True,
        )
        if not initialization["converged"]:
            warnings.warn(
                "MD-ASH exchangeable L-BFGS ended before meeting endpoint criteria; convergence is not established.",
                RuntimeWarning,
                stacklevel=2,
            )
    state = {
        "version": 1,
        "solver": "MD-ASH",
        "prior": "halfnormal",
        "kind": kind,
        "sign_intercept": sign_intercept,
        "scale": scale.clone(),
        "covariate_roles": roles,
        "feature_mean": mean.clone(),
        "feature_sd": sd.clone(),
        "fit_options": options,
    }
    slopes = y.new_empty(0)
    if kind == "exchangeable":
        if pi is None:
            pi = _exchangeable_weights(model_param, likelihood.shape[1])
        if initialization is not None:
            optimization = initialization
        state["pi"] = pi.detach().clone()
    else:
        model = _conditional_model(columns, likelihood.shape[1], model_param, pi, sign_intercept)
        if budget:
            if model_param is not None:
                continuation_start, model = _continuation_start(
                    model,
                    _nanstandardise(X, mean, sd),
                    likelihood,
                    options["penalty"],
                )
            optimization = _train_model(
                model,
                _nanstandardise(X, mean, sd),
                likelihood,
                budget,
                options["penalty"],
                verbose,
                "MD-ASH",
                options["coefficient_prior_scale"],
                options["coefficient_prior_df"],
            )
        if initialization is not None:
            initialization["conditional_uniform_mass"] = _INITIAL_UNIFORM_MASS
            optimization["initialization"] = initialization
        state["state_dict"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        slopes = model.w.detach()
    if continuation_start is not None:
        optimization["continuation_start"] = continuation_start
    state["optimization"] = optimization
    result = predict_md_ash(X, y, se, model_param=state)
    loss = -torch.logsumexp(likelihood + result.log_pi, 1).sum()
    regularization = (
        -(options["penalty"] - 1) * result.log_pi[:, 0].mean() if options["penalty"] != 1 else y.new_zeros(())
    )
    if options["coefficient_prior_scale"] is not None:
        prior = torch.distributions.StudentT(
            y.new_tensor(options["coefficient_prior_df"]),
            scale=options["coefficient_prior_scale"],
            validate_args=False,
        )
        regularization -= prior.log_prob(slopes).sum()
    _require_finite(loss, "MD-ASH loss")
    _require_finite(regularization, "MD-ASH regularization")
    if budget == 0:
        optimization["mean_objective"] = ((loss + regularization) / len(y)).item()
    if verbose:
        inactive = [i for i, value in enumerate(sd) if value == 0]
        print(f"[MD-ASH half-normal] {kind}; {sign_intercept} sign intercept; roles={roles}; inactive={inactive}")
        print(f"Penalty settings: {options}; endpoint criteria met: {optimization['converged']}")
    return cash_PosteriorMeanNorm(
        result.post_mean,
        result.post_mean2,
        result.post_sd,
        result.log_pi.exp(),
        scale.clone(),
        loss=loss,
        model_param=state,
        regularization_loss=regularization,
    )
