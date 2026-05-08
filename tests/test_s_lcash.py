"""Tests for S-LC-ASH (Scaled Linear Covariate-mediated Adaptive Shrinkage).

Covers kernels, parameter container, warm-start, panel fit, cold-start,
and a precision spot-check.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy import stats as sps

from cebmf_torch.cebnm import s_lc_ash_new_level_posterior_means, s_lc_ash_posterior_means
from cebmf_torch.cebnm.cash_solver import cash_PosteriorMeanNorm
from cebmf_torch.cebnm.s_lc_ash import (
    SLCAshNet,
    s_lc_ash_compute_posteriors,
    s_lc_ash_log_marginal,
    _warm_start_from_pooled_ash,
)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _scipy_log_marginal(beta, se, c, p, w, sigma):
    spike = p * sps.norm.pdf(beta, loc=0.0, scale=se)
    slab = (1 - p) * sum(
        wk * sps.norm.pdf(beta, loc=0.0, scale=math.sqrt(c**2 * sk**2 + se**2))
        for wk, sk in zip(w, sigma, strict=True)
    )
    return math.log(spike + slab)


def _scipy_post_moments(beta, se, c, p, w, sigma):
    log_spike = math.log(p) + sps.norm.logpdf(beta, loc=0.0, scale=se)
    log_slab = []
    m_slab, v_slab = [], []
    for wk, sk in zip(w, sigma, strict=True):
        var_g = c**2 * sk**2 + se**2
        log_slab.append(
            math.log(1 - p) + math.log(wk) + sps.norm.logpdf(beta, loc=0.0, scale=math.sqrt(var_g))
        )
        ratio = (c**2 * sk**2) / var_g
        m_slab.append(ratio * beta)
        v_slab.append(ratio * se**2)
    log_components = np.array([log_spike] + log_slab)
    log_marg = float(np.logaddexp.reduce(log_components))
    r = np.exp(log_components - log_marg)
    pmean = sum(r[k + 1] * m_slab[k] for k in range(len(w)))
    pmean2 = sum(r[k + 1] * (v_slab[k] + m_slab[k] ** 2) for k in range(len(w)))
    return log_marg, pmean, pmean2


def _simulate_panel(
    n_per_level=400,
    n_levels=8,
    *,
    c_per_level=None,
    p_global=0.7,
    slab_widths=(0.05, 0.2, 0.8),
    slab_weights=(0.5, 0.3, 0.2),
    se_value=0.1,
    seed=0,
):
    rng = np.random.default_rng(seed)
    if c_per_level is None:
        c_per_level = [1.0] * n_levels
    rows = []
    for t in range(n_levels):
        c_t = c_per_level[t]
        for _ in range(n_per_level):
            if rng.random() < p_global:
                beta_true = 0.0
            else:
                k = rng.choice(len(slab_widths), p=slab_weights)
                beta_true = rng.normal(0.0, c_t * slab_widths[k])
            beta_hat = beta_true + rng.normal(0.0, se_value)
            rows.append((beta_hat, se_value, t))
    return {
        "betahat": torch.tensor([r[0] for r in rows], dtype=torch.float64),
        "sebetahat": torch.tensor([r[1] for r in rows], dtype=torch.float64),
        "X_cat": torch.tensor([r[2] for r in rows], dtype=torch.long),
        "n_levels": n_levels,
        "c_per_level": np.array(c_per_level),
    }


# ---------------------------------------------------------------------------
# Marginal density kernel.
# ---------------------------------------------------------------------------


class TestMarginalDensity:
    def test_matches_scipy_handcrafted(self):
        T, K = 3, 4
        sigma = [0.05, 0.1, 0.3, 0.8]
        w = [0.5, 0.2, 0.2, 0.1]
        c_t_vals = [0.5, 1.0, 2.0]
        p = 0.85
        records = [
            (0, 0.01, 0.05), (0, -0.04, 0.05), (0, 0.0, 0.02),
            (1, 0.3, 0.1), (1, -0.5, 0.3), (1, 0.05, 0.15),
            (2, 1.5, 0.4), (2, -2.0, 0.3), (2, 0.0, 0.1),
        ]
        level_id = torch.tensor([r[0] for r in records], dtype=torch.long)
        beta = torch.tensor([r[1] for r in records], dtype=torch.float64)
        se = torch.tensor([r[2] for r in records], dtype=torch.float64)
        log_c = torch.log(torch.tensor(c_t_vals, dtype=torch.float64))
        logit_p = torch.tensor(math.log(p / (1 - p)), dtype=torch.float64)
        eta = torch.log(torch.tensor(w, dtype=torch.float64))
        sigma_t = torch.tensor(sigma, dtype=torch.float64)

        log_m = s_lc_ash_log_marginal(beta, se, level_id, log_c, logit_p, eta, sigma_t)
        for i, (tid, b, s) in enumerate(records):
            ref = _scipy_log_marginal(b, s, c_t_vals[tid], p, w, sigma)
            assert math.isclose(float(log_m[i]), ref, rel_tol=1e-9, abs_tol=1e-9)

    def test_finite_for_extreme_se(self):
        T, K = 1, 3
        sigma_t = torch.tensor([0.1, 0.5, 2.0], dtype=torch.float64)
        eta = torch.zeros(K, dtype=torch.float64)
        log_c = torch.zeros(T, dtype=torch.float64)
        logit_p = torch.tensor(0.0, dtype=torch.float64)
        beta = torch.tensor([1e-3, 0.0, 1.0, 1e-3, 0.0, 1.0], dtype=torch.float64)
        se = torch.tensor([1e-8, 1e-8, 1e-8, 1e8, 1e8, 1e8], dtype=torch.float64)
        level_id = torch.zeros_like(beta, dtype=torch.long)
        log_m = s_lc_ash_log_marginal(beta, se, level_id, log_c, logit_p, eta, sigma_t)
        assert torch.isfinite(log_m).all()

    def test_rejects_zero_sigma(self):
        sigma_t = torch.tensor([0.0, 0.5], dtype=torch.float64)
        eta = torch.zeros(2, dtype=torch.float64)
        log_c = torch.zeros(1, dtype=torch.float64)
        logit_p = torch.tensor(0.0, dtype=torch.float64)
        beta = torch.tensor([0.1], dtype=torch.float64)
        se = torch.tensor([0.1], dtype=torch.float64)
        level_id = torch.zeros(1, dtype=torch.long)
        with pytest.raises(ValueError, match="strictly positive"):
            s_lc_ash_log_marginal(beta, se, level_id, log_c, logit_p, eta, sigma_t)

    def test_gradient_flows(self):
        T, K = 2, 3
        sigma_t = torch.tensor([0.1, 0.5, 2.0], dtype=torch.float64)
        eta = torch.zeros(K, dtype=torch.float64, requires_grad=True)
        log_c = torch.zeros(T, dtype=torch.float64, requires_grad=True)
        logit_p = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        torch.manual_seed(1)
        beta = torch.randn(50, dtype=torch.float64) * 0.3
        se = torch.full((50,), 0.1, dtype=torch.float64)
        level_id = torch.randint(0, T, (50,))
        log_m = s_lc_ash_log_marginal(beta, se, level_id, log_c, logit_p, eta, sigma_t)
        (-log_m.sum()).backward()
        assert log_c.grad is not None and torch.any(log_c.grad != 0)
        assert logit_p.grad is not None and abs(float(logit_p.grad)) > 0
        assert eta.grad is not None and torch.any(eta.grad != 0)


# ---------------------------------------------------------------------------
# Posterior moments kernel.
# ---------------------------------------------------------------------------


class TestPosteriors:
    def test_matches_scipy(self):
        T, K = 2, 3
        sigma = [0.1, 0.4, 1.5]
        w = [0.6, 0.3, 0.1]
        c_t_vals = [1.0, 2.5]
        p = 0.85
        records = [
            (0, 0.05, 0.05), (0, -0.2, 0.1), (0, 1.0, 0.05),
            (1, 0.3, 0.1), (1, -1.5, 0.4), (1, 0.0, 0.05),
        ]
        level_id = torch.tensor([r[0] for r in records], dtype=torch.long)
        beta = torch.tensor([r[1] for r in records], dtype=torch.float64)
        se = torch.tensor([r[2] for r in records], dtype=torch.float64)
        log_c = torch.log(torch.tensor(c_t_vals, dtype=torch.float64))
        logit_p = torch.tensor(math.log(p / (1 - p)), dtype=torch.float64)
        eta = torch.log(torch.tensor(w, dtype=torch.float64))
        sigma_t = torch.tensor(sigma, dtype=torch.float64)
        out = s_lc_ash_compute_posteriors(beta, se, level_id, log_c, logit_p, eta, sigma_t)
        for i, (tid, b, s) in enumerate(records):
            lm, pm, pm2 = _scipy_post_moments(b, s, c_t_vals[tid], p, w, sigma)
            assert math.isclose(float(out["log_marginal"][i]), lm, rel_tol=1e-10, abs_tol=1e-12)
            assert math.isclose(float(out["post_mean"][i]), pm, rel_tol=1e-9, abs_tol=1e-12)
            assert math.isclose(float(out["post_mean2"][i]), pm2, rel_tol=1e-9, abs_tol=1e-12)

    def test_responsibilities_sum_to_one(self):
        torch.manual_seed(2)
        T, K = 3, 5
        sigma_t = torch.linspace(0.05, 2.0, K, dtype=torch.float64)
        eta = torch.randn(K, dtype=torch.float64) * 0.5
        log_c = torch.randn(T, dtype=torch.float64) * 0.3
        logit_p = torch.tensor(1.5, dtype=torch.float64)
        beta = torch.randn(60, dtype=torch.float64) * 0.5
        se = torch.full((60,), 0.1, dtype=torch.float64)
        level_id = torch.randint(0, T, (60,))
        out = s_lc_ash_compute_posteriors(beta, se, level_id, log_c, logit_p, eta, sigma_t)
        torch.testing.assert_close(out["pi_np"].sum(dim=1), torch.ones(60, dtype=torch.float64),
                                   rtol=1e-12, atol=1e-12)

    def test_spike_at_column_zero(self):
        T, K = 1, 3
        sigma_t = torch.tensor([0.1, 0.5, 2.0], dtype=torch.float64)
        eta = torch.zeros(K, dtype=torch.float64)
        log_c = torch.zeros(T, dtype=torch.float64)
        logit_p = torch.tensor(6.0, dtype=torch.float64)
        beta = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        se = torch.tensor([0.05, 0.05, 0.05], dtype=torch.float64)
        level_id = torch.zeros(3, dtype=torch.long)
        out = s_lc_ash_compute_posteriors(beta, se, level_id, log_c, logit_p, eta, sigma_t)
        assert (out["pi_np"][:, 0] > 0.9).all()
        assert out["pi_np"].shape == (3, K + 1)


# ---------------------------------------------------------------------------
# Container.
# ---------------------------------------------------------------------------


class TestSLCAshNet:
    def test_construct_and_invariants(self):
        T, K = 5, 4
        sigma = torch.tensor([0.05, 0.2, 0.8, 2.0], dtype=torch.float64)
        log_w = torch.zeros(K, dtype=torch.float64)
        net = SLCAshNet(T, sigma, log_w, logit_p_init=1.5, log_c_init=0.0)
        assert net.log_c.shape == (T,)
        assert net.logit_p.shape == ()
        assert net.eta.shape == (K,)
        assert net.sigma.shape == (K,)
        assert hasattr(net, "mu_c") and hasattr(net, "log_tau_c")
        torch.testing.assert_close(net.c(), torch.ones(T, dtype=torch.float64))
        # logit_p starts at 1.5 -> p ≈ 0.818
        assert math.isclose(float(net.p()), 1 / (1 + math.exp(-1.5)), abs_tol=1e-9)

    def test_rejects_zero_sigma(self):
        with pytest.raises(ValueError, match="strictly positive"):
            SLCAshNet(2, torch.tensor([0.0, 0.5]), torch.zeros(2), logit_p_init=0.0)

    def test_state_dict_round_trip(self):
        T, K = 3, 4
        sigma = torch.tensor([0.05, 0.2, 0.8, 2.0], dtype=torch.float64)
        log_w = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
        net = SLCAshNet(T, sigma, log_w, logit_p_init=0.5, log_c_init=0.1)
        state = net.state_dict()
        for k in ("sigma", "log_c", "logit_p", "eta", "mu_c", "log_tau_c"):
            assert k in state
        net2 = SLCAshNet(T, sigma, log_w, logit_p_init=0.0, log_c_init=0.0)
        net2.load_state_dict(state)
        torch.testing.assert_close(net.log_c, net2.log_c)
        torch.testing.assert_close(net.logit_p, net2.logit_p)

    def test_recentre_eta_preserves_softmax(self):
        T, K = 2, 4
        sigma = torch.tensor([0.05, 0.2, 0.8, 2.0], dtype=torch.float64)
        log_w = torch.tensor([1.5, -0.5, 2.0, -1.0], dtype=torch.float64)
        net = SLCAshNet(T, sigma, log_w, logit_p_init=0.0)
        w_before = net.w().clone()
        net.recentre_eta_()
        assert math.isclose(float(net.eta.mean()), 0.0, abs_tol=1e-12)
        torch.testing.assert_close(net.w(), w_before, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Warm start.
# ---------------------------------------------------------------------------


class TestWarmStart:
    def test_strips_spike_entry(self):
        rng = np.random.default_rng(10)
        N = 1000
        ses = np.full(N, 0.1)
        is_null = rng.random(N) < 0.7
        betas = np.where(is_null, 0.0, rng.normal(0, 0.5, N))
        betahat = torch.tensor(betas + rng.normal(0, ses, N), dtype=torch.float64)
        se = torch.tensor(ses, dtype=torch.float64)
        warm = _warm_start_from_pooled_ash(betahat, se, ash_init=False)
        assert torch.all(warm["sigma"] > 0)
        # log_w is finite and exponentiates to a probability vector summing to 1
        slab_w = torch.exp(warm["log_w"])
        torch.testing.assert_close(slab_w.sum(),
                                    torch.tensor(1.0, dtype=slab_w.dtype),
                                    rtol=1e-6, atol=1e-6)

    def test_clips_p_to_floor_ceiling(self):
        rng = np.random.default_rng(11)
        N = 2000
        ses = np.full(N, 0.1)
        betahat = torch.tensor(rng.normal(0, ses), dtype=torch.float64)
        se = torch.tensor(ses, dtype=torch.float64)
        warm = _warm_start_from_pooled_ash(betahat, se, ash_init=False,
                                            initial_p_floor=0.01, initial_p_ceiling=0.99)
        p_init = 1.0 / (1.0 + math.exp(-warm["logit_p_init"]))
        assert 0.01 - 1e-12 <= p_init <= 0.99 + 1e-12


# ---------------------------------------------------------------------------
# Panel fit.
# ---------------------------------------------------------------------------


class TestPanelFit:
    def test_returns_cash_posterior_mean_norm_with_expected_fields(self):
        sim = _simulate_panel(n_per_level=200, n_levels=4, seed=1)
        res = s_lc_ash_posterior_means(
            sim["betahat"], sim["sebetahat"], sim["X_cat"], sim["n_levels"],
            n_epochs=80, verbose=False, seed=1,
        )
        assert isinstance(res, cash_PosteriorMeanNorm)
        N = sim["betahat"].numel()
        assert res.post_mean.shape == (N,)
        assert res.pi_np.shape == (N, res.scale.numel() + 1)
        assert res._arch_meta["family"] == "s_lc_ash"
        assert res._arch_meta["n_levels"] == sim["n_levels"]
        psi = res.priors_fitted[0]
        for k in ("mu_c", "tau2_c", "p", "solver"):
            assert k in psi
        tp = res.level_params
        assert tp["c"].shape == (sim["n_levels"],)
        assert tp["p"].shape == (1,)
        assert math.isfinite(res.marginal_loglik)

    def test_recovers_known_c_correlated(self):
        torch.manual_seed(7)
        n_levels = 12
        rng = np.random.default_rng(7)
        true_c = np.exp(rng.normal(0.0, 0.7, size=n_levels))
        sim = _simulate_panel(
            n_per_level=600, n_levels=n_levels,
            c_per_level=true_c.tolist(), p_global=0.7, seed=7,
        )
        res = s_lc_ash_posterior_means(
            sim["betahat"], sim["sebetahat"], sim["X_cat"], n_levels,
            n_epochs=400, verbose=False, seed=7,
        )
        c_hat = res.level_params["c"].numpy()
        c_corr = float(np.corrcoef(np.log(c_hat), np.log(true_c))[0, 1])
        assert c_corr > 0.7, f"log-c correlation {c_corr:.3f} below threshold"

    def test_seed_reproducibility(self):
        sim = _simulate_panel(n_per_level=200, n_levels=4, seed=3)
        kw = dict(n_epochs=80, verbose=False, seed=99)
        res_a = s_lc_ash_posterior_means(sim["betahat"], sim["sebetahat"], sim["X_cat"], 4, **kw)
        res_b = s_lc_ash_posterior_means(sim["betahat"], sim["sebetahat"], sim["X_cat"], 4, **kw)
        torch.testing.assert_close(res_a.post_mean, res_b.post_mean, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(res_a.level_params["c"], res_b.level_params["c"],
                                   rtol=1e-12, atol=1e-12)

    def test_homogeneous_panel_collapses_tau2_c_to_floor(self):
        sim = _simulate_panel(
            n_per_level=800, n_levels=10,
            c_per_level=[1.0] * 10, p_global=0.7, seed=11,
        )
        res = s_lc_ash_posterior_means(
            sim["betahat"], sim["sebetahat"], sim["X_cat"], 10,
            n_epochs=400, tau2_min=1e-6, verbose=False, seed=11,
        )
        psi = res.priors_fitted[0]
        # Joint Adam stabilises tau (does not collapse to 1e-6 like alternating
        # EB does) but keeps it small on a homogeneous panel.
        assert psi["tau2_c"] < 0.05, f"tau2_c={psi['tau2_c']:.4g} unexpectedly large on homogeneous panel"

    def test_heterogeneous_panel_keeps_tau2_c_above_floor(self):
        sim = _simulate_panel(
            n_per_level=800, n_levels=10,
            c_per_level=[0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0],
            p_global=0.7, seed=12,
        )
        res = s_lc_ash_posterior_means(
            sim["betahat"], sim["sebetahat"], sim["X_cat"], 10,
            n_epochs=400, verbose=False, seed=12,
        )
        psi = res.priors_fitted[0]
        assert psi["tau2_c"] > 0.05

    def test_predict_pi_raises(self):
        sim = _simulate_panel(n_per_level=80, n_levels=2, seed=8)
        res = s_lc_ash_posterior_means(
            sim["betahat"], sim["sebetahat"], sim["X_cat"], 2, n_epochs=20, verbose=False,
        )
        with pytest.raises(NotImplementedError, match="s_lc_ash_new_level_posterior_means"):
            res.predict_pi(X_cat=torch.tensor([0, 1], dtype=torch.long))

    def test_state_dict_kernel_round_trip(self):
        sim = _simulate_panel(n_per_level=100, n_levels=3, seed=5)
        res = s_lc_ash_posterior_means(
            sim["betahat"], sim["sebetahat"], sim["X_cat"], 3,
            n_epochs=40, verbose=False, seed=5,
        )
        state = res.model_param
        K = res.scale.numel()
        net = SLCAshNet(3, res.scale, torch.zeros(K, dtype=torch.float64), logit_p_init=0.0)
        net.load_state_dict(state)
        with torch.no_grad():
            out = s_lc_ash_compute_posteriors(
                sim["betahat"], sim["sebetahat"], sim["X_cat"],
                net.log_c, net.logit_p, net.eta, net.sigma,
            )
        torch.testing.assert_close(out["post_mean"], res.post_mean, rtol=1e-9, atol=1e-12)

    def test_rejects_invalid_inputs(self):
        sim = _simulate_panel(n_per_level=20, n_levels=2, seed=8)
        with pytest.raises(ValueError, match="strictly positive"):
            s_lc_ash_posterior_means(
                sim["betahat"], -sim["sebetahat"], sim["X_cat"], 2, n_epochs=10, verbose=False,
            )
        with pytest.raises(ValueError, match="X_cat values"):
            bad = sim["X_cat"].clone()
            bad[0] = 99
            s_lc_ash_posterior_means(
                sim["betahat"], sim["sebetahat"], bad, 2, n_epochs=10, verbose=False,
            )

    def test_track_loglik_history_does_not_change_posteriors(self):
        sim = _simulate_panel(n_per_level=150, n_levels=3, seed=4)
        kw = dict(n_epochs=60, verbose=False, seed=4)
        res_off = s_lc_ash_posterior_means(
            sim["betahat"], sim["sebetahat"], sim["X_cat"], 3,
            track_loglik_history=False, **kw,
        )
        res_on = s_lc_ash_posterior_means(
            sim["betahat"], sim["sebetahat"], sim["X_cat"], 3,
            track_loglik_history=True, **kw,
        )
        torch.testing.assert_close(res_off.post_mean, res_on.post_mean, rtol=1e-12, atol=1e-12)
        last = res_on.priors_fitted_history[-1][0]
        assert "loglik_history" in last
        assert len(last["loglik_history"]) == 60
        for x in last["loglik_history"]:
            assert math.isfinite(x)


# ---------------------------------------------------------------------------
# Cold-start.
# ---------------------------------------------------------------------------


class TestNewLevelPosteriorMeans:
    def _fit_panel(self, **kw):
        sim = _simulate_panel(
            n_per_level=600, n_levels=10,
            c_per_level=[0.3, 0.5, 0.8, 1.2, 1.5, 2.0, 2.5, 3.0, 0.7, 1.8],
            p_global=0.7, seed=40,
        )
        res = s_lc_ash_posterior_means(
            sim["betahat"], sim["sebetahat"], sim["X_cat"], 10,
            n_epochs=400, verbose=False, seed=40, **kw,
        )
        return sim, res

    def test_returns_cash_posterior_mean_norm(self):
        sim, panel = self._fit_panel()
        mask = sim["X_cat"] == 0
        new = s_lc_ash_new_level_posterior_means(sim["betahat"][mask], sim["sebetahat"][mask], panel,
                             n_epochs=200, verbose=False, seed=40)
        assert isinstance(new, cash_PosteriorMeanNorm)
        assert new.post_mean.shape == sim["betahat"][mask].shape
        assert new._arch_meta["family"] == "s_lc_ash"
        assert new._arch_meta["single_level"] is True
        assert new.level_params["c"].shape == (1,)

    def test_does_not_mutate_panel(self):
        sim, panel = self._fit_panel()
        c_before = panel.level_params["c"].clone()
        psi_before = dict(panel.priors_fitted[0])
        mask = sim["X_cat"] == 3
        _ = s_lc_ash_new_level_posterior_means(sim["betahat"][mask], sim["sebetahat"][mask], panel,
                            n_epochs=100, verbose=False, seed=40)
        torch.testing.assert_close(panel.level_params["c"], c_before, rtol=0, atol=0)
        for k, v in psi_before.items():
            assert panel.priors_fitted[0][k] == v

    def test_freezes_layer_b(self):
        sim, panel = self._fit_panel()
        mask = sim["X_cat"] == 5
        new = s_lc_ash_new_level_posterior_means(sim["betahat"][mask], sim["sebetahat"][mask], panel,
                             n_epochs=100, verbose=False, seed=40)
        torch.testing.assert_close(new.scale, panel.scale, rtol=0, atol=0)
        torch.testing.assert_close(new.model_param["eta"], panel.model_param["eta"],
                                   rtol=0, atol=0)
        torch.testing.assert_close(new.model_param["logit_p"], panel.model_param["logit_p"],
                                   rtol=0, atol=0)
        psi_new = new.priors_fitted[0]
        psi_panel = panel.priors_fitted[0]
        assert psi_new["mu_c"] == psi_panel["mu_c"]
        assert psi_new["frozen_from_panel"] is True

    def test_round_trip_recovers_panel(self):
        sim, panel = self._fit_panel()
        c_panel = panel.level_params["c"].numpy()
        c_cold = []
        for t in range(sim["n_levels"]):
            mask = sim["X_cat"] == t
            new = s_lc_ash_new_level_posterior_means(sim["betahat"][mask], sim["sebetahat"][mask], panel,
                                 n_epochs=300, verbose=False, seed=40)
            c_cold.append(float(new.level_params["c"][0]))
        c_cold = np.array(c_cold)
        log_c_diff = np.abs(np.log(c_cold) - np.log(c_panel))
        assert log_c_diff.max() < 0.10, f"max log-c diff = {log_c_diff.max():.4f}"

    def test_tau_inflate_widens_prior(self):
        sim, panel = self._fit_panel()
        c_panel = panel.level_params["c"].numpy()
        log_c = np.log(c_panel)
        mu_c = float(panel.priors_fitted[0]["mu_c"])
        t_extreme = int(np.argmax(np.abs(log_c - mu_c)))
        mask = sim["X_cat"] == t_extreme
        b, s = sim["betahat"][mask], sim["sebetahat"][mask]
        new1 = s_lc_ash_new_level_posterior_means(b, s, panel, tau_inflate=1.0, n_epochs=300, verbose=False, seed=40)
        new3 = s_lc_ash_new_level_posterior_means(b, s, panel, tau_inflate=3.0, n_epochs=300, verbose=False, seed=40)
        d1 = abs(math.log(float(new1.level_params["c"][0])) - mu_c)
        d3 = abs(math.log(float(new3.level_params["c"][0])) - mu_c)
        assert d3 >= d1 - 1e-3

    def test_rejects_non_s_lc_ash_panel(self):
        bad = cash_PosteriorMeanNorm(
            post_mean=torch.zeros(3), post_mean2=torch.zeros(3), post_sd=torch.zeros(3),
            pi_np=torch.zeros(3, 4), scale=torch.tensor([0.1, 0.2, 0.5]),
            _arch_meta={"family": "lcash"},
        )
        with pytest.raises(ValueError, match="s_lc_ash_posterior_means"):
            s_lc_ash_new_level_posterior_means(torch.zeros(3), torch.ones(3), bad)

    def test_single_observation_level(self):
        sim, panel = self._fit_panel()
        b = sim["betahat"][:1]
        s = sim["sebetahat"][:1]
        new = s_lc_ash_new_level_posterior_means(b, s, panel, n_epochs=100, verbose=False, seed=42)
        assert math.isfinite(new.marginal_loglik)
        c_hat = float(new.level_params["c"][0])
        mu_c = float(panel.priors_fitted[0]["mu_c"])
        tau_c = math.sqrt(panel.priors_fitted[0]["tau2_c"])
        # With only 1 observation, the cold-start sits within ~3 prior SDs of mu_c.
        assert abs(math.log(c_hat) - mu_c) <= 3.0 * max(tau_c, 0.1)

    def test_predict_pi_raises_for_single_level(self):
        sim, panel = self._fit_panel()
        mask = sim["X_cat"] == 0
        new = s_lc_ash_new_level_posterior_means(sim["betahat"][mask], sim["sebetahat"][mask], panel,
                             n_epochs=20, verbose=False)
        with pytest.raises(NotImplementedError, match="s_lc_ash_new_level_posterior_means"):
            new.predict_pi(X_cat=torch.tensor([0], dtype=torch.long))


# ---------------------------------------------------------------------------
# Float32 spot-check.
# ---------------------------------------------------------------------------


class TestFloat32:
    def test_kernel_float32_close_to_float64(self):
        torch.manual_seed(50)
        T, K = 3, 4
        sigma_t64 = torch.tensor([0.05, 0.2, 0.8, 2.0], dtype=torch.float64)
        eta64 = torch.randn(K, dtype=torch.float64) * 0.3
        log_c64 = torch.randn(T, dtype=torch.float64) * 0.4
        logit_p64 = torch.tensor(1.5, dtype=torch.float64)
        beta64 = torch.randn(50, dtype=torch.float64) * 0.4
        se64 = torch.full((50,), 0.1, dtype=torch.float64)
        level_id = torch.randint(0, T, (50,))

        log_m_64 = s_lc_ash_log_marginal(beta64, se64, level_id, log_c64, logit_p64, eta64, sigma_t64)
        log_m_32 = s_lc_ash_log_marginal(
            beta64.float(), se64.float(), level_id,
            log_c64.float(), logit_p64.float(), eta64.float(), sigma_t64.float(),
        )
        torch.testing.assert_close(log_m_32.double(), log_m_64, rtol=1e-4, atol=1e-4)
