"""MD-ASH through cEBMF: repeated updates, missing cells, roles, and pruning."""

import pytest
import torch

from cebmf_torch import cEBMF
from cebmf_torch.cebnm import md_ash_posterior_means, predict_md_ash


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("mode", ["shared", "component"])
@pytest.mark.parametrize("side", ["L", "F"])
def test_repeated_factor_updates_use_roles_and_finite_posteriors(mode, side):
    g = torch.Generator().manual_seed(125)
    x = torch.randn(32, 2, generator=g, dtype=torch.float64)
    loading = x[:, 0] + 0.3 * torch.randn(32, generator=g, dtype=torch.float64)
    factor = torch.randn(24, generator=g, dtype=torch.float64)
    y = loading[:, None] * factor[None, :] + 0.5 * torch.randn(32, 24, generator=g, dtype=torch.float64)
    y[0, :] = float("nan")
    y[1::7, ::4] = float("nan")
    if side == "F":
        y = y.T
    model = cEBMF(
        y,
        K=2,
        S=0.5,
        allow_backfitting=False,
        device="cpu",
        **{
            f"prior_{side}": "md_ash",
            f"X_{side.lower()}": x,
            f"prior_{side}_kwargs": {
                "covariate_roles": ["direction", "magnitude"],
                "sign_intercept": mode,
                "scale": [0, 0.25, 0.75, 2.0],
                "max_iter": 100,
            },
            "self_row_cov" if side == "L" else "self_col_cov": True,
        },
    )
    model.initialise_factors()
    model.fit(4)
    for array in (model.L, model.F, model.L2, model.F2, torch.tensor(model.obj)):
        assert torch.isfinite(array).all()
    assert (model.L2 >= model.L.square()).all()
    assert (model.F2 >= model.F.square()).all()
    assert max(b - a for a, b in zip(model.obj, model.obj[1:], strict=False)) < 1e-3
    states = getattr(model, f"model_state_{side}")
    assert states[0]["covariate_roles"] == ("direction", "magnitude")
    assert states[1]["covariate_roles"] == ("direction", "magnitude", "magnitude")
    assert all(state["sign_intercept"] == mode for state in states)
    assert all("continuation_start" in state["optimization"] for state in states)
    # Remove the first factor. Earlier-factor covariates and cached states change.
    model.pi0_L = [1.0, 0.0]
    model.pi0_F = [0.0, 0.0]
    model.model.allow_backfitting = True
    model._backfit()
    model.model.allow_backfitting = False
    assert model.model.K == 1
    model.fit(2)
    state = getattr(model, f"model_state_{side}")[0]
    assert state["covariate_roles"] == ("direction", "magnitude")
    assert torch.isfinite(model.L).all() and torch.isfinite(model.F).all()


def test_cpu_scope_is_explicit():
    with pytest.raises(ValueError, match="CPU fitting"):
        md_ash_posterior_means(None, [1.0], [1.0], device="cuda")


def test_numpy_inputs_match_tensor_inputs():
    import numpy as np

    y = np.linspace(-2, 2, 24)
    s = np.ones_like(y)
    actual = md_ash_posterior_means(None, y, s, scale=[0, 0.5, 2])
    expected = md_ash_posterior_means(None, torch.from_numpy(y), torch.from_numpy(s), scale=[0, 0.5, 2])
    torch.testing.assert_close(actual.post_mean, expected.post_mean, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["shared", "component"])
@pytest.mark.parametrize("side", ["L", "F"])
@pytest.mark.parametrize("known_se", [True, False])
def test_unobserved_matrix_rows_do_not_change_prior_fit(mode, side, known_se):
    x = torch.linspace(-2, 2, 80, dtype=torch.float64)[:, None]
    y = 2 * x[:, 0] + torch.randn(80, generator=torch.Generator().manual_seed(31), dtype=torch.float64)
    fitted = []
    for extra in (0, 80):
        xx = torch.cat((x, x.new_full((extra, 1), 100)))
        yy = torch.cat((y, y.new_full((extra,), float("nan"))))[:, None]
        if side == "F":
            yy = yy.T
        model = cEBMF(
            yy,
            K=1,
            S=1.0 if known_se else None,
            device="cpu",
            allow_backfitting=False,
            **{
                f"prior_{side}": "md_ash",
                f"X_{side.lower()}": xx,
                f"prior_{side}_kwargs": {
                    "scale": [0, 0.5, 2],
                    "covariate_roles": "both",
                    "sign_intercept": mode,
                    "max_iter": 500,
                },
            },
        )
        other = "F" if side == "L" else "L"
        getattr(model, other).fill_(1)
        getattr(model, other + "2").fill_(1)
        getattr(model, f"_update_{side}_factor")(0, model.Y0.clone(), model.tau_map if known_se else None, 1e-12)
        state = getattr(model, f"model_state_{side}")[0]
        torch.testing.assert_close(state["feature_mean"], x.mean(0), rtol=0, atol=0)
        fitted.append(model)
        if extra:
            prior = predict_md_ash(xx, model_param=state)
            torch.testing.assert_close(getattr(model, side)[80:, 0], prior.post_mean[80:], rtol=0, atol=0)
            torch.testing.assert_close(getattr(model, side + "2")[80:, 0], prior.post_mean2[80:], rtol=0, atol=0)
    a, b = fitted
    torch.testing.assert_close(getattr(a, side)[:, 0], getattr(b, side)[:80, 0], rtol=0, atol=0)
    torch.testing.assert_close(getattr(a, f"kl_{side.lower()}"), getattr(b, f"kl_{side.lower()}"))
    torch.testing.assert_close(
        getattr(a, f"regularization_{side.lower()}"), getattr(b, f"regularization_{side.lower()}"), rtol=0, atol=0
    )


@pytest.mark.parametrize("initialization", ["zero", "svd"])
@pytest.mark.parametrize("prune", [True, False])
@pytest.mark.parametrize("n_factors", [1, 2])
def test_zero_factor_precision_deactivates_before_pruning(initialization, prune, n_factors):
    y = 0.1 * torch.randn(20, 10, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    model = cEBMF(
        y,
        K=n_factors,
        S=1.0,
        device="cpu",
        allow_backfitting=prune,
        prior_L="md_ash",
        prior_F="md_ash",
        prior_L_kwargs={"scale": [0]},
        prior_F_kwargs={"scale": [0, 1]},
    )
    model.initialise_factors(initialization)
    model.fit(3)
    assert torch.isfinite(torch.tensor(model.obj)).all()
    assert model.model.K == (1 if prune else n_factors)
    torch.testing.assert_close(model.L @ model.F.T, torch.zeros_like(y))
    assert model.kl_l.sum() == model.kl_f.sum() == 0
