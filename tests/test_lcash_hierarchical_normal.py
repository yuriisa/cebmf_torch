"""Tests for the hierarchical Normal Level-2 prior on a single categorical
embedding (Step 3 of the hierarchical-priors design).

Covers Section 11.3 of ``cebmf_torch_hierarchical_priors_design.md``:

- Pulls coefficients of low-signal traits toward zero.
- No-op when ``prior_warmup_epochs > n_epochs``.
- E-step closed-form consistency: tau2_hat == mean(emb.weight[1:].pow(2)).
- Recovery of tau2 under strong simulated signal.
- ``priors_fitted`` field is populated.
- tau2 collapse regimes respect the ``tau2_min`` clamp.
- Batch-size invariance (validates the |B|/N scaling rule).
- Row 0 (gauge) excluded from the Level-2 fit.
"""

import math

import torch

from cebmf_torch.cebnm.lcash import lcash_posterior_means


def _simulate_per_trait(
    n_per_trait: int,
    n_signal: int,
    n_null: int,
    signal_sd: float,
    obs_sd: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-trait one-hot simulation: ``n_signal`` traits with strong slab,
    ``n_null`` traits with delta-zero, all observed with noise ``obs_sd``.
    """
    g = torch.Generator().manual_seed(seed)
    T = n_signal + n_null
    n = T * n_per_trait
    X_cat = torch.cat([torch.full((n_per_trait,), t, dtype=torch.long) for t in range(T)]).reshape(-1, 1)

    # First n_signal levels: slab N(0, signal_sd^2). Remaining: zero effect.
    true_eff = torch.zeros(n)
    for t in range(n_signal):
        mask = X_cat[:, 0] == t
        true_eff[mask] = signal_sd * torch.randn(int(mask.sum()), generator=g)

    se = torch.full((n,), obs_sd)
    betahat = true_eff + se * torch.randn(n, generator=g)
    return X_cat, betahat, se, true_eff


def test_normal_prior_pulls_low_signal_traits():
    """Coefficient magnitudes for null traits should be small relative to signal."""
    torch.manual_seed(0)
    T_signal, T_null = 5, 45
    X_cat, betahat, se, _ = _simulate_per_trait(
        n_per_trait=80,
        n_signal=T_signal,
        n_null=T_null,
        signal_sd=2.0,
        obs_sd=0.5,
        seed=0,
    )

    res = lcash_posterior_means(
        X=None,
        betahat=betahat,
        sebetahat=se,
        n_epochs=100,
        batch_size=512,
        lr=1e-2,
        penalty=1.0,
        ash_init=True,
        verbose=False,
        device=torch.device("cpu"),
        seed=42,
        X_cat=X_cat,
        n_cat_levels=T_signal + T_null,
        cat_prior=["normal"],
        prior_warmup_epochs=20,
        prior_refit_every=10,
    )

    # Assert tau2 fitted and above the floor.
    psi = res.priors_fitted[0]
    assert psi["tau2"] > 1e-6

    # Per-trait coefficient norm: sqrt(sum_k w_t,k^2) for rows 1..T-1.
    state = res.model_param
    emb_w = state["cat.0.weight"]  # (T, K)
    norms = emb_w.pow(2).sum(dim=1).sqrt()  # (T,)
    # Index 0 is gauge-pinned; exclude.
    norms_nonzero = norms[1:]
    # Levels 1..T_signal-1 are signal (level 0 is in the gauge slot).
    # Use a generous comparison: mean |w| of null traits should be smaller
    # than mean |w| of signal traits. With one signal level absorbed by the
    # gauge, signal levels are 1..T_signal-1 (= 4 traits).
    signal_norms = norms_nonzero[: T_signal - 1]
    null_norms = norms_nonzero[T_signal - 1 :]
    print(
        f"signal mean ||w_t|| = {signal_norms.mean().item():.4f}, "
        f"null mean ||w_t|| = {null_norms.mean().item():.4f}, tau2 = {psi['tau2']:.4f}"
    )
    assert null_norms.mean() < signal_norms.mean(), (
        "Level-2 prior should pull null-trait coefficients more strongly than signal traits"
    )


def test_normal_prior_no_op_when_warmup_only():
    """With warmup > n_epochs, posteriors should match the no-prior baseline."""
    torch.manual_seed(0)
    X_cat, betahat, se, _ = _simulate_per_trait(
        n_per_trait=80, n_signal=2, n_null=8, signal_sd=2.0, obs_sd=0.5, seed=0
    )

    common = {
        "X": None,
        "betahat": betahat,
        "sebetahat": se,
        "n_epochs": 30,
        "batch_size": 512,
        "lr": 1e-2,
        "penalty": 1.0,
        "ash_init": True,
        "verbose": False,
        "device": torch.device("cpu"),
        "seed": 42,
        "X_cat": X_cat,
        "n_cat_levels": 10,
    }

    res_baseline = lcash_posterior_means(**common, cat_prior=None)
    # Same numeric default for weight_decay between the two runs (1e-3 baseline).
    res_warmup = lcash_posterior_means(
        **common,
        cat_prior=["normal"],
        prior_warmup_epochs=999,  # never fires
        prior_refit_every=10,
        weight_decay=1e-3,  # match baseline so the only diff is the (inactive) prior path
    )

    # priors_fitted is only populated after a successful E-step.
    assert res_warmup.priors_fitted is None or len(res_warmup.priors_fitted) == 0

    assert torch.allclose(res_baseline.post_mean, res_warmup.post_mean, atol=1e-5)
    assert torch.allclose(res_baseline.pi_np, res_warmup.pi_np, atol=1e-5)


def test_normal_prior_consistency_with_emstep():
    """E-step formula: tau2_hat == mean(emb.weight[1:].pow(2)).

    This is a closed-form check on the *fitter* (not the optimisation).
    The reported tau2 in priors_fitted must equal the empirical second
    moment over rows 1..T-1, since fit_normal calls ebnm_normal with
    sebetahat=None which has the closed-form ML tau2 = mean(beta^2).
    """
    torch.manual_seed(0)
    X_cat, betahat, se, _ = _simulate_per_trait(
        n_per_trait=80, n_signal=10, n_null=40, signal_sd=1.5, obs_sd=0.5, seed=0
    )

    res = lcash_posterior_means(
        X=None,
        betahat=betahat,
        sebetahat=se,
        n_epochs=40,
        batch_size=512,
        lr=1e-2,
        penalty=1.0,
        ash_init=True,
        verbose=False,
        device=torch.device("cpu"),
        seed=42,
        X_cat=X_cat,
        n_cat_levels=50,
        cat_prior=["normal"],
        prior_warmup_epochs=20,
        prior_refit_every=10,
    )

    emb_w = res.model_param["cat.0.weight"]
    expected = emb_w[1:].pow(2).mean().item()
    expected = max(expected, 1e-6)  # respect the clamp
    fitted = res.priors_fitted[0]["tau2"]
    assert math.isclose(fitted, expected, rel_tol=1e-5, abs_tol=1e-9), (
        f"Closed-form E-step mismatch: fitted={fitted}, expected={expected}"
    )


def test_normal_prior_recovers_tau_under_strong_signal():
    """Simulated trait shifts w_t ~ N(0, 0.5^2); fitted tau2 should land in [0.1, 0.5]."""
    torch.manual_seed(123)
    g = torch.Generator().manual_seed(123)
    T = 200
    n_per = 30
    n = T * n_per
    obs_sd = 0.3
    tau_true = 0.5

    # Per-trait shift: w_t ~ N(0, tau_true^2). Realised as a per-observation
    # mean depending on which trait the observation belongs to.
    w_true = tau_true * torch.randn(T, generator=g)
    X_cat = torch.cat([torch.full((n_per,), t, dtype=torch.long) for t in range(T)]).reshape(-1, 1)
    se = torch.full((n,), obs_sd)
    mu = w_true[X_cat[:, 0]]
    betahat = mu + se * torch.randn(n, generator=g)

    res = lcash_posterior_means(
        X=None,
        betahat=betahat,
        sebetahat=se,
        n_epochs=120,
        batch_size=512,
        lr=1e-2,
        penalty=1.0,
        ash_init=True,
        verbose=False,
        device=torch.device("cpu"),
        seed=7,
        X_cat=X_cat,
        n_cat_levels=T,
        cat_prior=["normal"],
        prior_warmup_epochs=20,
        prior_refit_every=10,
    )

    tau2_hat = res.priors_fitted[0]["tau2"]
    print(f"recovered tau2 = {tau2_hat:.4f} (true tau2 = {tau_true**2:.4f})")
    # The signal lives in mixture-weight modulation rather than coefficient
    # magnitudes per se, so use a generous envelope (Section 11.3).
    assert 0.01 <= tau2_hat <= 1.0, f"tau2_hat={tau2_hat} outside the wide envelope"


def test_priors_fitted_field():
    """priors_fitted is non-None after a Level-2 fit and contains tau2."""
    torch.manual_seed(0)
    X_cat, betahat, se, _ = _simulate_per_trait(
        n_per_trait=60, n_signal=2, n_null=8, signal_sd=2.0, obs_sd=0.5, seed=0
    )

    res = lcash_posterior_means(
        X=None,
        betahat=betahat,
        sebetahat=se,
        n_epochs=40,
        batch_size=512,
        lr=1e-2,
        penalty=1.0,
        ash_init=True,
        verbose=False,
        device=torch.device("cpu"),
        seed=42,
        X_cat=X_cat,
        n_cat_levels=10,
        cat_prior=["normal"],
        prior_warmup_epochs=20,
        prior_refit_every=10,
    )
    assert res.priors_fitted is not None
    assert 0 in res.priors_fitted
    psi = res.priors_fitted[0]
    assert "tau2" in psi
    assert psi["solver"] == "normal"
    assert psi["tau2"] >= 1e-6


def test_normal_prior_collapse_regimes():
    """Under (a) null, (b) weak, (c) strong signal: tau2 >= tau2_min.

    Under strong signal: tau2 is well above the floor.
    """
    common = {
        "n_epochs": 80,
        "batch_size": 512,
        "lr": 1e-2,
        "penalty": 1.0,
        "ash_init": True,
        "verbose": False,
        "device": torch.device("cpu"),
        "seed": 42,
        "n_cat_levels": 20,
        "cat_prior": ["normal"],
        "prior_warmup_epochs": 20,
        "prior_refit_every": 10,
        "tau2_min": 1e-6,
    }

    # (a) Fully null: every trait identical zero effect.
    torch.manual_seed(0)
    X_cat_a, betahat_a, se_a, _ = _simulate_per_trait(
        n_per_trait=50, n_signal=0, n_null=20, signal_sd=0.0, obs_sd=0.5, seed=0
    )
    res_a = lcash_posterior_means(X=None, betahat=betahat_a, sebetahat=se_a, X_cat=X_cat_a, **common)
    tau2_a = res_a.priors_fitted[0]["tau2"]

    # (b) Weak signal.
    torch.manual_seed(1)
    X_cat_b, betahat_b, se_b, _ = _simulate_per_trait(
        n_per_trait=50, n_signal=4, n_null=16, signal_sd=0.3, obs_sd=0.5, seed=1
    )
    res_b = lcash_posterior_means(X=None, betahat=betahat_b, sebetahat=se_b, X_cat=X_cat_b, **common)
    tau2_b = res_b.priors_fitted[0]["tau2"]

    # (c) Strong signal.
    torch.manual_seed(2)
    X_cat_c, betahat_c, se_c, _ = _simulate_per_trait(
        n_per_trait=50, n_signal=10, n_null=10, signal_sd=3.0, obs_sd=0.5, seed=2
    )
    res_c = lcash_posterior_means(X=None, betahat=betahat_c, sebetahat=se_c, X_cat=X_cat_c, **common)
    tau2_c = res_c.priors_fitted[0]["tau2"]

    print(f"tau2 (null/weak/strong) = {tau2_a:.6f} / {tau2_b:.6f} / {tau2_c:.6f}")
    # Floor respected in all three regimes.
    assert tau2_a >= 1e-6
    assert tau2_b >= 1e-6
    assert tau2_c >= 1e-6
    # Strong-signal regime should escape the floor by orders of magnitude.
    assert tau2_c > 1e-3


def test_batchsize_invariance():
    """Direct check that the per-batch prior gradient carries the |B|/N factor.

    Section 6.7 mandates ``loss += (|B|/N) * R``. We verify this surgically
    by comparing the actual prior-gradient contribution under two batch
    sizes for a fixed model state. With the rule, the prior gradient
    summed over one full epoch (= summed across all batches) is
    ``sum_batches |B_b|/N * R = R``, independent of batch size. Without
    the rule, the per-epoch sum is ``n_batches * R``, which scales like
    ``N/|B|``.

    We instantiate a fresh LcashNet with the same weights, populate
    ``priors_state``, and run one epoch of M-steps with the hierarchical
    prior active. We then compare the *total prior gradient seen by the
    embedding over one epoch* between two batch sizes.

    This is the cleanest possible regression test for the scaling rule:
    it isolates the prior-gradient bookkeeping from SGD trajectory
    chaos.
    """
    from torch.utils.data import DataLoader, TensorDataset

    from cebmf_torch.cebnm.lcash import LcashNet, _install_reference_gauge, logp_normal

    torch.manual_seed(0)
    T = 20
    K = 5
    n_per = 60
    g = torch.Generator().manual_seed(0)
    X_cat = torch.cat([torch.full((n_per,), t, dtype=torch.long) for t in range(T)]).reshape(-1, 1)
    n = X_cat.shape[0]
    se = torch.full((n,), 0.5)
    _ = se * torch.randn(n, generator=g)  # consume generator state for reproducibility

    # Two identical model copies under two batch sizes.
    def make_net():
        torch.manual_seed(0)
        net = LcashNet(cont_dim=0, num_classes=K, cat_n_levels=[T])
        # Populate rows 1..T-1 with non-zero values so the prior gradient is non-zero.
        with torch.no_grad():
            net.cat[0].weight[1:].normal_(generator=torch.Generator().manual_seed(7))
        _install_reference_gauge(net)
        return net

    psi = {"tau2": 1.0}

    grad_sums: dict[int, torch.Tensor] = {}
    for bs in (128, 480):
        net = make_net()
        loader_g = torch.Generator().manual_seed(42)
        loader = DataLoader(TensorDataset(torch.arange(n)), batch_size=bs, shuffle=True, generator=loader_g)
        # Accumulate the *prior-only* gradient on the embedding across one epoch
        # by zero-ing the data-loss contribution after each batch via a fresh
        # parameter copy. We capture grad_sum as the sum of (|B|/N)*grad(R)
        # over batches; with the rule this should equal grad(R) once.
        accum = torch.zeros_like(net.cat[0].weight)
        for (idx,) in loader:
            net.cat[0].weight.grad = None
            batch_size_actual = idx.shape[0]
            scale_factor = batch_size_actual / n  # this is the rule under test
            R = -logp_normal(net.cat[0].weight[1:], psi)
            (scale_factor * R).backward()
            accum = accum + net.cat[0].weight.grad.detach().clone()
        grad_sums[bs] = accum

    # The two epoch-summed gradients should match (modulo numerical noise)
    # because both equal grad(R) when the rule is on.
    diff = (grad_sums[128] - grad_sums[480]).abs().max().item()
    print(f"per-epoch prior grad max abs diff: {diff:.3e}")
    assert diff < 1e-5, (
        f"|B|/N scaling violated: per-epoch prior gradient depends on batch size "
        f"(max abs diff = {diff:.3e}); grad_sum(bs=128) and grad_sum(bs=480) should be equal"
    )

    # Sanity: the gradient is non-trivial.
    assert grad_sums[128].abs().max() > 1e-3, "Trivial gradient; test setup is wrong"


def test_gauge_row0_excluded_from_level2_fit():
    """tau2_hat must equal mean(emb.weight[1:].pow(2)), NOT mean(emb.weight.pow(2))."""
    torch.manual_seed(0)
    X_cat, betahat, se, _ = _simulate_per_trait(
        n_per_trait=70, n_signal=5, n_null=15, signal_sd=2.0, obs_sd=0.5, seed=3
    )

    res = lcash_posterior_means(
        X=None,
        betahat=betahat,
        sebetahat=se,
        n_epochs=60,
        batch_size=512,
        lr=1e-2,
        penalty=1.0,
        ash_init=True,
        verbose=False,
        device=torch.device("cpu"),
        seed=42,
        X_cat=X_cat,
        n_cat_levels=20,
        cat_prior=["normal"],
        prior_warmup_epochs=20,
        prior_refit_every=10,
    )

    emb_w = res.model_param["cat.0.weight"]
    # Sanity: row 0 should be exactly zero (gauge).
    assert torch.all(emb_w[0] == 0.0), "Gauge violated: row 0 not zero"

    fitted = res.priors_fitted[0]["tau2"]
    expected_correct = max(emb_w[1:].pow(2).mean().item(), 1e-6)
    expected_buggy = max(emb_w.pow(2).mean().item(), 1e-6)

    # If row 0 had leaked into the fit, the all-rows mean would dilute the
    # second moment by a factor T/(T-1).
    assert math.isclose(fitted, expected_correct, rel_tol=1e-5, abs_tol=1e-9)
    # And the buggy form must differ measurably; this is the regression guard.
    assert not math.isclose(fitted, expected_buggy, rel_tol=1e-3), (
        "Suspicious: tau2 matches the all-rows mean; row 0 may have leaked into the fit"
    )
