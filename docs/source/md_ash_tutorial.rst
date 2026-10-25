MD-ASH tutorial: predict every supplied gene
================================================

This tutorial fits loss-of-function (LoF) effects with side information.
It uses the public ``cebmf_torch`` interface and synthetic data.
The example teaches data preparation and prediction. It does not establish biological accuracy or probability calibration.

See :doc:`md_ash` for the model, parameters, penalties, and numerical limits.
MD-ASH currently uses CPU float64.

Run the example
-------------------

From a checkout that includes MD-ASH, create an environment and install the package:

.. code-block:: sh

   python3 -m venv .venv
   .venv/bin/python -m pip install -e . matplotlib
   .venv/bin/python examples/md_ash.py --output md-ash-example --plot

The script writes four files:

* ``predictions.csv`` contains one row for each of the 1,000 supplied genes.
* ``fit.pt`` stores fitted parameters, imputation means, column names, and roles.
* ``summary.json`` records populations, predictive scores, and optimizer diagnostics.
* ``posterior_means.png`` compares predictions with and without side information.

The plot separates usable source observations from genes that receive only their conditional prior.
Both fits use the shared sign intercept and the same component standard deviations.
Neither fit uses target observations during fitting.
Warnings and saved diagnostics remain part of the result. A completed script does not establish convergence.

To run the displayed sections interactively, first import the public functions and set the seed and dimensions:

.. literalinclude:: ../../examples/md_ash.py
   :language: python
   :start-after: # BEGIN public_imports
   :end-before: # END public_imports

.. literalinclude:: ../../examples/md_ash.py
   :language: python
   :start-after: # BEGIN setup
   :end-before: # END setup
   :dedent: 4

1. Keep one row per gene
----------------------------

Start from the full protein-coding gene list for a real analysis.
Align source effects, target effects, counts, and covariates to that list by gene identifier.
This example uses 1,000 synthetic rows to illustrate the same contract.

``betahat`` is the source estimate. ``sebetahat`` is its standard error, in the same effect units.
The observation model is ``betahat[i] ~ Normal(beta[i], sebetahat[i]**2)``.
``target`` is an independent estimate of the same effect. Its standard error is ``target_se``.

The source mask selects finite estimates, positive finite standard errors, and minor allele count (MAC) of at least six.
Unknown counts pass this example's count filter. The target mask applies its own conditions independently.
Keep a finite source estimate in the data even when its count fails. Set its source mask to false.

.. literalinclude:: ../../examples/md_ash.py
   :language: python
   :start-after: # BEGIN observations
   :end-before: # END observations
   :dedent: 4

2. Prepare covariates using source fitting rows
---------------------------------------------------

``s_het`` represents the GeneBayes v3 posterior mean in a real analysis.
``Z_mis`` is a signed missense burden-test statistic from the source panel.
Here both are synthetic. The example uses these columns and roles:

.. list-table:: Supplied columns and their use by MD-ASH
   :header-rows: 1

   * - Column
     - Role
     - Meaning
   * - ``s_het``
     - magnitude
     - Change weights across component standard deviations.
   * - ``abs(Z_mis)``
     - magnitude
     - Use the strength of missense evidence, regardless of sign.
   * - ``Z_mis``
     - direction
     - Change the probability of a positive effect within each pair.
   * - ``s_het_missing``
     - magnitude
     - Allow a separate magnitude shift when s_het is missing.
   * - ``missense_missing``
     - both
     - Allow separate magnitude and direction shifts when missense evidence is missing.

Construct the absolute value before imputation. Estimate each term's mean from source fitting rows with that term observed.
Use those means for every prediction row. Do not use target outcomes or target eligibility to estimate them.
MD-ASH then standardizes the supplied columns internally, including indicators.

.. literalinclude:: ../../examples/md_ash.py
   :language: python
   :start-after: # BEGIN covariates
   :end-before: # END covariates
   :dedent: 4

The coefficients can have either sign. The roles do not impose biological sign constraints.
Adding a direction role to the missing indicator permits a mean shift among genes with missing missense results.
It therefore changes more than the signed-Z slope alone.

3. Fit the source rows, then predict all rows
-------------------------------------------------

``X`` has shape ``(n, p)``. Effects, standard errors, and masks have shape ``(n,)``.
Fit only the selected source rows. Pass all rows to prediction, with their source mask.
A missing or excluded source observation contributes no likelihood. Its posterior equals its fitted conditional prior.

.. literalinclude:: ../../examples/md_ash.py
   :language: python
   :start-after: # BEGIN fit_predict
   :end-before: # END fit_predict
   :dedent: 4

The default point-mass penalty is 10.
The regression slopes have a Student-t prior with coefficient-prior scale 1 and five degrees of freedom.
The solver selects component standard deviations from source observations when ``scale`` is omitted.
These component standard deviations differ from the Student-t coefficient-prior scale.

Use ``sign_intercept="component"`` in a fresh fit to fit a separate sign intercept for each component pair.
Keep the complete returned ``model_param`` for later prediction. Save the external preparation settings with it.

4. Score independent targets
--------------------------------

Scored genes need valid target observations. They do not need a usable source observation.
Mean log predictive density (LPD) measures prediction of target estimates. Larger values are better for the same target observations.
It convolves the full posterior mixture with target observation error analytically.

.. literalinclude:: ../../examples/md_ash.py
   :language: python
   :start-after: # BEGIN score
   :end-before: # END score
   :dedent: 4

Posterior means and sign probabilities describe the latent effect.
LPD describes a noisy target estimate. A gain in one summary need not imply a gain in the other.

5. Use MD-ASH in matrix factorization
-----------------------------------------

The script also fits a small gene-by-trait matrix through ``cEBMF``.
``X_l`` supplies row covariates. ``prior_L_kwargs`` assigns their roles.
This interface demonstration supplies the known simulated loading as its covariate. The true loading would normally be unknown.
The known observation standard error is 0.2. The returned loading and factor matrices have 40 and 12 rows, respectively.

.. literalinclude:: ../../examples/md_ash.py
   :language: python
   :start-after: # BEGIN matrix
   :end-before: # END matrix
   :dedent: 4

The first matrix row has no observations. Its loading receives the conditional prior.
Use the default singular-value decomposition (SVD) initialization. Zero initialization cannot activate these MD-ASH factors.
This four-update example checks the interface. It does not establish matrix-factorization convergence.
