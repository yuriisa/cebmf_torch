Magnitude-and-direction adaptive shrinkage
==========================================

Magnitude-and-direction adaptive shrinkage (MD-ASH) fits a conditional prior for noisy effect estimates. Covariates can
change effect magnitude, direction, or both. The observation model is
``betahat[i] ~ Normal(beta[i], sebetahat[i]**2)``.

The prior has a point mass at zero and paired negative and positive half-normal
components. The component standard deviations are fixed before fitting.
The solver estimates their weights through two linear covariate models.
All posterior moments and predictive log densities retain the full mixture.

For a complete source-to-target example, see :doc:`md_ash_tutorial`.

Model
-----

Let :math:`i` index observations and :math:`k=1,\ldots,K` index positive
component standard deviations :math:`a_k`. Write :math:`a_0=0` for the point
mass. Supplied ``scale`` contains :math:`[0,a_1,\ldots,a_K]`.
Each :math:`a_k` is the standard deviation of a normal distribution before
truncation, not a coefficient-prior scale.

For standardized magnitude covariates :math:`x_i` and fitted slopes
:math:`b`, proportional odds defines

.. math::

   F_{ik}=\operatorname{logistic}(c_k-x_i^T b),\quad k=0,\ldots,K-1,

   q_{i0}=F_{i0},\quad q_{ik}=F_{ik}-F_{i,k-1}\ (0<k<K),\quad
   q_{iK}=1-F_{i,K-1}.

The :math:`K` fitted cutpoints :math:`c_k` are ordered. Larger
:math:`x_i^T b` moves weight toward larger component standard deviations.

For standardized direction covariates :math:`z_i` and fitted slopes :math:`d`,

.. math::

   r_{ik}=\operatorname{logistic}(\alpha_k+z_i^T d),\quad
   \pi_{ik}^{+}=q_{ik}r_{ik},\quad
   \pi_{ik}^{-}=q_{ik}(1-r_{ik}).

The prior weights are :math:`q_{i0}` at zero and :math:`\pi_{ik}^{-},
\pi_{ik}^{+}` on each half-normal pair. Returned components have order
``zero, negative_a1, positive_a1, negative_a2, positive_a2, ...``.

``sign_intercept="shared"`` is the fresh-fit default. It fits one
:math:`\alpha` across component standard deviations. ``"component"`` fits
one :math:`\alpha_k` for each pair. Both can fit a nonzero prior mean without
covariates. Only the component option permits unrestricted half-normal
mixture weights without covariates. If all direction logits are zero,
the prior becomes the symmetric normal-mixture PO-LC-ASH model.

Inputs and penalties
--------------------

``covariate_roles`` accepts ``"magnitude"``, ``"direction"``, ``"both"``,
or one such value per input column. The default is magnitude.
A column with role ``"both"`` has separate magnitude and direction slopes.

Prepare missing covariates externally. Estimate imputation means from the
source rows used for fitting. Add missing indicators and apply the same
preparation to prediction rows. The solver requires finite input columns.
It standardizes each active column internally, including missing indicators.
It saves those means and standard deviations for prediction and continuation.
Constant training columns remain inactive.

The solver does not take absolute values or squares implicitly. For missense
burden information, supply ``abs(Z_mis)`` with role ``"magnitude"`` and
``Z_mis`` with role ``"direction"``. Supply the missing indicator separately.

The default point-mass penalty is 10. The slope prior is Student-t with
5 degrees of freedom and coefficient-prior scale 1. These settings match
PO-LC-ASH. Cutpoints and sign intercepts do not receive the slope penalty.
``coefficient_prior_scale=None`` disables that penalty. These penalties do
not guarantee conservative or calibrated posterior probabilities.

Fit and predict
---------------

Use ``cebmf_torch.cebnm.md_ash_posterior_means`` on eligible source rows.
Use ``cebmf_torch.cebnm.predict_md_ash`` on the full prediction population.
For a row without a source observation, the posterior equals its conditional
prior. Pass ``observed=False`` for that row, or omit source arrays for a
prior-only prediction. Do not replace missing source effects with observed
zeroes.

Both functions use CPU float64. MD-ASH does not currently support GPU fitting.
The fit returns posterior means, second moments, standard deviations,
prior weights, separate likelihood and penalty losses, and ``model_param``.
Prediction also returns component posterior weights and sign probabilities.

Here is a complete example with one covariate and 50 prediction rows.
``X`` has shape ``(n, p)``. Effects and standard errors have shape ``(n,)``.
The Boolean ``observed`` mask also has shape ``(n,)``.
This example has no missing covariates. Missing source effects remain unobserved.

.. code-block:: python

   import torch
   from cebmf_torch.cebnm import md_ash_posterior_means, predict_md_ash

   rng = torch.Generator().manual_seed(12)
   X = torch.linspace(-1, 1, 50, dtype=torch.float64)[:, None]
   y = X[:, 0] + torch.randn(50, generator=rng, dtype=torch.float64)
   se = torch.ones_like(y)
   y[-8:] = float("nan")
   observed = torch.isfinite(y)
   fit = md_ash_posterior_means(
       X[observed], y[observed], se[observed], covariate_roles="both",
   )
   prediction = predict_md_ash(
       X, y, se, observed=observed, model_param=fit.model_param,
   )
   print(prediction.post_mean, prediction.post_sd)
   print(prediction.prob_negative, prediction.prob_zero, prediction.prob_positive)

Each printed attribute has one value per prediction row, including all eight
rows without source observations. ``fit.pi_np`` contains prior component
weights for fitting rows. ``prediction.posterior_weights`` contains posterior
component weights for all prediction rows, with shape ``(n, 2*K+1)``.

For predictive log density, pass independent target effects and their standard
errors. Select only rows with valid target data. All supplied arrays must
refer to those same rows. For example:

.. code-block:: python

   target = X[:, 0] + torch.randn(50, generator=rng, dtype=torch.float64)
   target_se = torch.ones_like(target)
   scored = torch.arange(50) % 5 == 0
   score = predict_md_ash(
       X[scored], y[scored], se[scored], observed=observed[scored],
       target=target[scored], target_se=target_se[scored],
       model_param=fit.model_param,
   )
   print(score.log_predictive)  # One log predictive density per scored row.

Target arrays must be finite, with positive standard errors. They never
enter fitting. Prediction convolves each posterior component with the target
observation error analytically, then sums the mixture densities in log space.

Keep the complete ``model_param`` dictionary. It includes component standard
deviations, covariate roles, standardization statistics, coefficients, penalty
settings, and optimizer diagnostics. Do not pass only its ``state_dict``.
Omitted settings inherit the saved values. Change roles, component standard
deviations, or sign-intercept mode through a fresh fit.

``max_iter=0`` requires saved state and evaluates it without fitting.
Its current convergence status is unassessed. Its objective uses the current
inputs and penalties. ``optimization["previous_fit"]`` retains the original
fit diagnostics, which do not establish convergence on changed inputs.
Positive-iteration continuation compares the saved start, an interior start,
and their sign reflections on the new observations. It fits only the best
start. The interior start adds 1% equal weight across component labels at
zero standardized covariates and retains the slopes. This prevents the
observed failures after sign reversal or emergence of larger effects.
It is an initialization rule, not a fitted probability floor.
Convergence warnings remain visible. Small gradients do not establish a
global optimum for conditional fits.

Matrix factorization
--------------------

Select ``prior_L="md_ash"`` or ``prior_F="md_ash"`` and ``device="cpu"``.
Pass solver options through ``prior_L_kwargs`` or ``prior_F_kwargs``.
For example, use ``{"covariate_roles": ["magnitude", "direction"]}`` for
two supplied covariate columns. With self-covariates enabled, appended earlier
factors use the magnitude role. Factor removal resets affected saved states.

Rows with zero likelihood precision do not enter fitting, standardization,
or penalties. They receive their fitted conditional prior, with zero KL
contribution. If an entire factor side has zero likelihood precision, cEBMF
sets that side to zero and clears its saved state. Normal factor pruning
can then remove it. Zero initialization therefore cannot activate MD-ASH
factors. Use the default SVD initialization to learn nonzero factors.

The solver reuses the package's proportional-odds model and L-BFGS objective.
Each factor update creates fresh L-BFGS curvature history. An explicit
``max_iter`` controls its budget. Otherwise the builder uses twice
``internal_epoch``, as for LC-ASH and PO-LC-ASH.

Validation
----------

Tests compare the half-normal likelihoods and moments with independent
integration and saved ashr references. They check symmetric normal-mixture
reduction, missing-source prediction, both sign-intercept options, and
saved-state replay. Repeated factorization tests cover both matrix sides,
missing cells, earlier-factor covariates, and factor removal.
