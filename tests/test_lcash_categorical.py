"""Tests for native categorical-covariate support in LC-ASH.

Covers Section 10 of ``cebmf_torch_hierarchical_priors_design.md``:
- Per-level prior recovery (categorical-only fit).
- Mixed continuous + categorical fit.
- Network-level equivalence between one-hot continuous and embedding.
- Validation: dtype, range, level counts.
- Reference-category gauge invariant (row 0 of every embedding stays at zero).
- Smoke test for the proportional-odds parameterisation with categorical input.
"""

import pytest
import torch

from cebmf_torch.cebnm.lcash import (
    LcashNet,
    _install_reference_gauge,
    lcash_posterior_means,
    po_lcash_posterior_means,
)


def _simulate_per_level(
    n_per_level: int,
    null_fractions: list[float],
    slab_sd: float = 2.0,
    obs_sd: float = 0.5,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[float]]:
    """Simulate categorical mixture data: each level has its own null fraction.

    Returns
    -------
    X_cat : long tensor (N, 1)
    betahat : float tensor (N,)
    sebetahat : float tensor (N,)
    null_fractions : the requested per-level null fractions (passed through)
    """
    g = torch.Generator().manual_seed(seed)
    X_cat = []
    betas = []
    for t, frac_null in enumerate(null_fractions):
        cat = torch.full((n_per_level,), t, dtype=torch.long)
        is_null = torch.rand(n_per_level, generator=g) < frac_null
        true_eff = torch.where(
            is_null,
            torch.zeros(n_per_level),
            slab_sd * torch.randn(n_per_level, generator=g),
        )
        X_cat.append(cat)
        betas.append(true_eff)
    X_cat = torch.cat(X_cat).reshape(-1, 1)
    true_beta = torch.cat(betas)
    se = torch.full((true_beta.shape[0],), obs_sd)
    betahat = true_beta + se * torch.randn(true_beta.shape[0], generator=g)
    return X_cat, betahat, se, null_fractions


def test_categorical_only_recovers_per_level_mixture():
    """Per-level fitted pi_0 should track the simulated null fractions."""
    torch.manual_seed(0)
    n_per_level = 2000
    null_fractions = [0.9, 0.5, 0.1]
    X_cat, betahat, se, _ = _simulate_per_level(n_per_level, null_fractions, seed=0)

    res = lcash_posterior_means(
        X=None,
        betahat=betahat,
        sebetahat=se,
        n_epochs=200,
        batch_size=512,
        lr=1e-2,
        penalty=1.0,
        ash_init=True,
        verbose=False,
        device=torch.device("cpu"),
        seed=42,
        X_cat=X_cat,
        n_cat_levels=len(null_fractions),
    )

    pi = res.pi_np  # (N, K)
    fitted_null_per_level = []
    for t in range(len(null_fractions)):
        mask = X_cat[:, 0] == t
        # All rows with the same categorical level share the same pi vector,
        # so the mean is exact (up to numerical noise).
        fitted_null_per_level.append(float(pi[mask, 0].mean().item()))

    fitted = torch.tensor(fitted_null_per_level)
    truth = torch.tensor(null_fractions)
    err = (fitted - truth).abs()
    print(f"per-level fitted pi_0: {fitted.tolist()}; truth: {truth.tolist()}")
    # The fit shrinks pi_0 toward the across-level average; require either
    # tight per-level recovery OR a strong monotone correlation across levels.
    if err.max() > 0.05:
        # Fall back to Spearman-rank check (robust given small T = 3).
        rank_fit = torch.argsort(torch.argsort(fitted))
        rank_true = torch.argsort(torch.argsort(truth))
        assert torch.equal(rank_fit, rank_true), (
            f"Per-level pi_0 recovery failed: fitted={fitted.tolist()}, truth={truth.tolist()}, abs_err={err.tolist()}"
        )


def test_continuous_plus_categorical():
    """Both continuous and categorical heads should drive non-trivial modulation."""
    torch.manual_seed(1)
    n_per_level = 1500
    n_levels = 3

    g = torch.Generator().manual_seed(1)
    X_cat = torch.cat([torch.full((n_per_level,), t, dtype=torch.long) for t in range(n_levels)]).reshape(-1, 1)
    n = X_cat.shape[0]
    # Continuous covariate: signal_sd grows with x, on top of the per-level base.
    x_cont = torch.randn(n, generator=g)
    # Per-level base null fraction.
    base_null = torch.tensor([0.85, 0.5, 0.15])[X_cat[:, 0]]
    # Continuous modulation: high x reduces the null fraction further.
    null_p = torch.clamp(base_null - 0.3 * torch.clamp(x_cont, min=0.0), min=0.05, max=0.95)
    is_null = torch.rand(n, generator=g) < null_p
    true_eff = torch.where(is_null, torch.zeros(n), 2.0 * torch.randn(n, generator=g))
    se = torch.full((n,), 0.5)
    betahat = true_eff + se * torch.randn(n, generator=g)

    res = lcash_posterior_means(
        X=x_cont.reshape(-1, 1),
        betahat=betahat,
        sebetahat=se,
        n_epochs=200,
        batch_size=512,
        lr=1e-2,
        penalty=1.0,
        ash_init=True,
        verbose=False,
        device=torch.device("cpu"),
        seed=42,
        X_cat=X_cat,
        n_cat_levels=n_levels,
    )

    pi = res.pi_np
    pi0 = pi[:, 0]

    # 1. Per-level pi_0 differs across categorical levels.
    per_level_means = torch.stack([pi0[X_cat[:, 0] == t].mean() for t in range(n_levels)])
    print(f"per-level pi_0 means: {per_level_means.tolist()}")
    assert per_level_means.max() - per_level_means.min() > 0.1, (
        "Categorical level should modulate pi_0 by at least 0.1"
    )

    # 2. Within a single level, continuous covariate modulates pi_0.
    mid_level = X_cat[:, 0] == 1
    x_in_level = x_cont[mid_level]
    pi0_in_level = pi0[mid_level]
    p25, p75 = torch.quantile(x_in_level, torch.tensor([0.25, 0.75]))
    low_mask = x_in_level <= p25
    high_mask = x_in_level >= p75
    print(
        f"within-level pi_0: low-x mean = {pi0_in_level[low_mask].mean():.3f}, "
        f"high-x mean = {pi0_in_level[high_mask].mean():.3f}"
    )
    assert pi0_in_level[low_mask].mean() > pi0_in_level[high_mask].mean(), (
        "Within a categorical level, low-x observations should have higher null fraction"
    )

    # 3. Sanity: the model returned valid posteriors (finite, bounded pi).
    assert torch.isfinite(res.post_mean).all()
    assert torch.isfinite(res.pi_np).all()
    assert torch.allclose(res.pi_np.sum(dim=1), torch.ones(n), atol=1e-4)


def test_lcashnet_one_hot_equivalence():
    """One-hot continuous LcashNet equals an embedding-based LcashNet."""
    torch.manual_seed(0)
    T, K = 4, 5
    N = 50

    # (a) cont_dim=T, no embedding, one-hot float input.
    g_a = torch.Generator().manual_seed(0)
    net_a = LcashNet(cont_dim=T, num_classes=K, generator=g_a)

    # (b) cont_dim=0, single embedding table of T levels.
    g_b = torch.Generator().manual_seed(0)
    net_b = LcashNet(cont_dim=0, num_classes=K, cat_n_levels=[T], generator=g_b)
    _install_reference_gauge(net_b)

    # Copy emb weight (T, K) into cont weight (K, T).
    with torch.no_grad():
        net_a.cont.weight.copy_(net_b.cat[0].weight.T)
        net_a.bias.copy_(net_b.bias)

    # Build matched inputs.
    indices = torch.randint(0, T, (N,), generator=g_a)
    x_cat = indices.reshape(-1, 1)
    x_one_hot = torch.zeros(N, T)
    x_one_hot[torch.arange(N), indices] = 1.0

    pi_a = net_a(x_one_hot, None)
    pi_b = net_b(None, x_cat)

    assert pi_a.shape == pi_b.shape == (N, K)
    assert torch.allclose(pi_a, pi_b, atol=1e-6, rtol=0), (
        f"One-hot vs embedding pi disagree, max abs diff = {(pi_a - pi_b).abs().max().item()}"
    )


def test_categorical_validates_dtype():
    """Float X_cat must raise TypeError mentioning 'long' and the X hint."""
    n = 50
    X_cat_bad = torch.randn(n, 1)  # float
    betahat = torch.randn(n)
    sebetahat = torch.ones(n)

    with pytest.raises(TypeError) as exc:
        lcash_posterior_means(
            X=None,
            betahat=betahat,
            sebetahat=sebetahat,
            n_epochs=5,
            verbose=False,
            device=torch.device("cpu"),
            X_cat=X_cat_bad,
            n_cat_levels=3,
        )
    msg = str(exc.value)
    assert "long" in msg.lower()
    assert "X" in msg  # the hint to use X for continuous data


def test_categorical_validates_range():
    """X_cat with index >= n_cat_levels must raise ValueError before training."""
    n = 50
    X_cat = torch.randint(0, 5, (n, 1), dtype=torch.long)
    X_cat[0, 0] = 5  # out of range when n_cat_levels = 5
    betahat = torch.randn(n)
    sebetahat = torch.ones(n)

    with pytest.raises(ValueError) as exc:
        lcash_posterior_means(
            X=None,
            betahat=betahat,
            sebetahat=sebetahat,
            n_epochs=5,
            verbose=False,
            device=torch.device("cpu"),
            X_cat=X_cat,
            n_cat_levels=5,
        )
    msg = str(exc.value).lower()
    assert "index" in msg or "n_cat_levels" in msg


def test_categorical_validates_degenerate_levels():
    """n_cat_levels < 2 must raise with 'degenerate' in the message."""
    n = 50
    X_cat = torch.zeros(n, 1, dtype=torch.long)
    betahat = torch.randn(n)
    sebetahat = torch.ones(n)

    with pytest.raises(ValueError) as exc:
        lcash_posterior_means(
            X=None,
            betahat=betahat,
            sebetahat=sebetahat,
            n_epochs=5,
            verbose=False,
            device=torch.device("cpu"),
            X_cat=X_cat,
            n_cat_levels=1,
        )
    assert "degenerate" in str(exc.value).lower()


@pytest.mark.parametrize("scenario", ["one_step", "many_steps", "warm_start"])
def test_categorical_gauge_row0_invariant(scenario):
    """emb.weight[0] must be zero after training under every code path."""
    torch.manual_seed(0)
    T, K = 4, 5
    N = 64

    net = LcashNet(cont_dim=0, num_classes=K, cat_n_levels=[T])
    if scenario == "warm_start":
        # Pretend a warm-start populated row 0 with non-zero values.
        with torch.no_grad():
            net.cat[0].weight.copy_(torch.randn_like(net.cat[0].weight))
    _install_reference_gauge(net)
    assert torch.all(net.cat[0].weight[0] == 0), "Gauge install did not zero row 0"

    optim = torch.optim.Adam(net.parameters(), lr=1e-2)
    n_steps = 1 if scenario == "one_step" else 100

    for _ in range(n_steps):
        x_cat = torch.randint(0, T, (N, 1))
        pi = net(None, x_cat)
        # Arbitrary scalar loss with non-zero gradient on row 0 (in the
        # absence of the gauge).  Cross-entropy against random targets.
        target = torch.randint(0, K, (N,))
        loss = torch.nn.functional.nll_loss(torch.log(pi.clamp_min(1e-30)), target)
        optim.zero_grad()
        loss.backward()
        optim.step()
        assert torch.all(net.cat[0].weight[0] == 0), f"Gauge violated after step in scenario={scenario}"


def test_propodds_categorical_basic():
    """po_lcash_posterior_means accepts X_cat and produces finite posteriors."""
    torch.manual_seed(2)
    n_per_level = 800
    null_fractions = [0.8, 0.4, 0.1]
    X_cat, betahat, se, _ = _simulate_per_level(n_per_level, null_fractions, seed=2)

    res = po_lcash_posterior_means(
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
        n_cat_levels=len(null_fractions),
    )

    n = X_cat.shape[0]
    assert res.pi_np.shape[0] == n
    assert torch.isfinite(res.post_mean).all()
    assert torch.isfinite(res.post_mean2).all()
    assert torch.isfinite(res.pi_np).all()
    assert torch.allclose(res.pi_np.sum(dim=1), torch.ones(n), atol=1e-4)
    assert torch.isfinite(torch.tensor(res.loss))
