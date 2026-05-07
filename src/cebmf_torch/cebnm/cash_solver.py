import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from cebmf_torch.utils.distribution_operation import get_data_loglik_normal_torch
from cebmf_torch.utils.mixture import autoselect_scales_mix_norm
from cebmf_torch.utils.posterior import posterior_mean_norm
from cebmf_torch.utils.standard_scaler import standard_scale


# ---- Dataset: assumes tensors already on correct device/dtype
class DensityRegressionDataset(Dataset):
    def __init__(self, X: torch.Tensor, betahat: torch.Tensor, sebetahat: torch.Tensor):
        self.X = X
        self.betahat = betahat
        self.sebetahat = sebetahat

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.betahat[idx], self.sebetahat[idx]


class CashNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, n_layers):
        """
        Initialize a neural network for CASH (Covariate Adaptive Shrinkage).

        Parameters
        ----------
        input_dim : int
            Number of input features.
        hidden_dim : int
            Number of hidden units in each layer.
        num_classes : int
            Number of mixture components (output classes).
        n_layers : int
            Number of hidden layers.
        """
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass through the CASH network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, input_dim).

        Returns
        -------
        torch.Tensor
            Mixture weights for each observation, shape (N, num_classes).
        """
        x = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        return self.softmax(self.output_layer(x))


# Custom loss function
def pen_loglik_loss(pred_pi, marginal_log_lik, penalty=1.5, epsilon=1e-10):
    L_batch = torch.exp(marginal_log_lik)  # (B, K)
    inner_sum = torch.sum(pred_pi * L_batch, dim=1)  # (B,)
    inner_sum = torch.clamp(inner_sum, min=epsilon)
    first_sum = torch.sum(torch.log(inner_sum))  # scalar

    if penalty > 1:
        # penalize per-gene spike probability (Dirichlet-like prior on component 0)
        log_pi0 = torch.log(torch.clamp(pred_pi[:, 0], min=epsilon))
        penalized_log_likelihood_value = first_sum + (penalty - 1) * torch.sum(log_pi0)
    else:
        penalized_log_likelihood_value = first_sum

    return -penalized_log_likelihood_value


class cash_PosteriorMeanNorm:
    def __init__(
        self,
        post_mean,
        post_mean2,
        post_sd,
        pi_np,
        scale,
        loss=0,
        model_param=None,
        priors_fitted=None,
        priors_fitted_history=None,
        marginal_loglik: float | None = None,
        x_means: torch.Tensor | None = None,
        x_stds: torch.Tensor | None = None,
    ):
        """
        Container for the results of the CASH posterior mean estimation.

        Parameters
        ----------
        post_mean : torch.Tensor
            Posterior means for each observation.
        post_mean2 : torch.Tensor
            Posterior second moments for each observation.
        post_sd : torch.Tensor
            Posterior standard deviations for each observation.
        pi_np : torch.Tensor
            Mixture weights for each observation.
        scale : torch.Tensor
            Mixture component scales.
        loss : float, optional
            Final training loss or log-likelihood.
        model_param : dict, optional
            Trained model parameters (state_dict).
        priors_fitted : dict or None, optional
            Fitted Level-2 hyperparameters keyed by categorical column
            (Step 3: integer keys, e.g.
            ``{0: {"tau2": 0.42, "solver": "normal"}}``). ``None`` when no
            Level-2 prior was active. Step 4 will move to string keys via
            the typed feature API.
        priors_fitted_history : list of dict or None, optional
            Per-E-step history of fitted Level-2 hyperparameters, one dict per
            E-step. Each dict has the same shape as ``priors_fitted``
            (column-index keys, ``{"tau2": ..., "solver": ...}`` values).
            ``None`` when no Level-2 prior was active. Useful for diagnosing
            whether ``tau2`` has stabilised by the end of training; users
            interpreting ``tau2`` for inferential purposes should verify it
            has flattened across the last several E-steps.
        marginal_loglik : float or None, optional
            The full-data marginal log-likelihood under the fitted prior,
            without the spike Dirichlet penalty. Numerically identical to
            ``-loss`` for cebmf_torch's LC-ASH path; exposed as a separate,
            explicitly-named field for users who track convergence or
            compare across hyperparameter settings without having to invert
            the sign on ``loss``.
        x_means : torch.Tensor or None, optional
            Per-column means of the continuous covariate matrix used at
            training time, computed NaN-aware. Cached so that
            :meth:`predict_pi` can apply the same standardisation to new
            ``X``. ``None`` when no continuous covariates were given.
        x_stds : torch.Tensor or None, optional
            Per-column standard deviations of the continuous covariate
            matrix used at training time, computed NaN-aware. Cached so
            that :meth:`predict_pi` can apply the same standardisation to
            new ``X``. ``None`` when no continuous covariates were given.
        """
        self.post_mean = post_mean
        self.post_mean2 = post_mean2
        self.post_sd = post_sd
        self.pi_np = pi_np
        self.loss = loss
        self.scale = scale
        self.model_param = model_param
        self.priors_fitted = priors_fitted
        self.priors_fitted_history = priors_fitted_history
        self.marginal_loglik = marginal_loglik
        self._x_means = x_means
        self._x_stds = x_stds

    def predict_pi(
        self,
        X: torch.Tensor | None = None,
        X_cat: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Compute prior mixture weights pi at new (X, X_cat) inputs.

        Reconstructs the trained network from ``self.model_param`` and runs
        a forward pass on the given inputs. Returns ``(N_new, K)`` mixture
        weights.

        Continuous covariates ``X`` are standardised using means and
        standard deviations cached from training (``self._x_means``,
        ``self._x_stds``). Pre-standardisation is not required.

        At least one of ``X`` and ``X_cat`` must be provided; whichever
        was used at training time should be used here.

        Parameters
        ----------
        X : torch.Tensor or None
            Continuous covariates, shape ``(N_new, F)``. Must be provided
            iff the trained model has a continuous head. NaN-aware
            standardisation is applied internally using the cached
            training statistics; pre-standardisation is not required.
        X_cat : torch.Tensor or None
            Categorical covariate indices, shape ``(N_new,)`` or
            ``(N_new, F_d)``, ``dtype=torch.long``. Must be provided iff
            the trained model has at least one categorical head. Each
            column ``d`` must have indices in
            ``[0, n_cat_levels[d])``.
        device : torch.device or None
            Compute device. Defaults to CPU when not given.

        Returns
        -------
        pi : torch.Tensor
            Per-observation mixture weights, shape ``(N_new, K)``. Rows
            sum to one within numerical tolerance.

        Raises
        ------
        ValueError
            If ``self.model_param`` is None (model was never trained), if
            both ``X`` and ``X_cat`` are None, or if the (X, X_cat)
            combination does not match the trained architecture.
            Also raised when ``X`` is given but no standardisation
            statistics were cached on this object.
        """
        # Local imports to avoid circular import at module load time.
        from cebmf_torch.cebnm.lcash import (
            LcashNet,
            PropOddsLcashNet,
            _apply_nanstandardise,
            _validate_and_normalise_cat,
        )

        if self.model_param is None:
            raise ValueError("predict_pi requires a trained model: self.model_param is None.")
        if X is None and X_cat is None:
            raise ValueError("predict_pi requires at least one of X, X_cat.")

        device = device or torch.device("cpu")
        state = self.model_param

        # Detect architecture: PropOddsLcashNet exposes ``delta_1``;
        # LcashNet exposes ``bias`` (and never ``delta_1``).
        is_po = "delta_1" in state

        # Determine cont_dim from cont.weight (LcashNet) or w (PO).
        if is_po:
            cont_dim = state["w"].shape[0] if "w" in state else 0
        else:
            cont_dim = state["cont.weight"].shape[1] if "cont.weight" in state else 0

        # Determine cat_n_levels by collecting cat.{d}.weight rows.
        cat_levels: list[int] = []
        d = 0
        while f"cat.{d}.weight" in state:
            cat_levels.append(state[f"cat.{d}.weight"].shape[0])
            d += 1

        # Determine K from self.scale.
        K = int(self.scale.shape[0])

        has_cont = cont_dim > 0
        has_cat = len(cat_levels) > 0

        # Validate the (X, X_cat) match against the trained architecture.
        if has_cont and not has_cat:
            if X is None:
                raise ValueError("Trained model has only a continuous head; X is required (X_cat must be None).")
            if X_cat is not None:
                raise ValueError("Trained model has only a continuous head; X_cat is not allowed (must be None).")
        elif has_cat and not has_cont:
            if X_cat is None:
                raise ValueError("Trained model has only a categorical head; X_cat is required (X must be None).")
            if X is not None:
                raise ValueError("Trained model has only a categorical head; X is not allowed (must be None).")
        elif has_cont and has_cat:
            if X is None or X_cat is None:
                raise ValueError(
                    "Trained model has both continuous and categorical heads; both X and X_cat must be provided."
                )
        else:
            # No heads at all is not a recoverable state.
            raise ValueError("Trained model has neither a continuous nor a categorical head; cannot predict.")

        # Reconstruct network. Pass the constructor's expected shapes; we
        # do not need log_pi_init or generator since load_state_dict will
        # overwrite the parameters.
        if is_po:
            net = PropOddsLcashNet(
                cont_dim=cont_dim,
                num_classes=K,
                cat_n_levels=cat_levels if cat_levels else None,
            )
        else:
            net = LcashNet(
                cont_dim=cont_dim,
                num_classes=K,
                cat_n_levels=cat_levels if cat_levels else None,
            )
        net.load_state_dict(state)
        net = net.to(device)
        net.eval()

        # Prepare X (NaN-aware standardisation using cached training stats).
        x_cont: torch.Tensor | None = None
        if X is not None:
            if self._x_means is None or self._x_stds is None:
                raise ValueError(
                    "predict_pi cannot standardise X: training did not save standardisation stats "
                    "(self._x_means / self._x_stds is None). Refit with the latest version, "
                    "or pre-standardise X yourself and use the cached internals."
                )
            X_t = torch.as_tensor(X, dtype=torch.float32)
            if X_t.ndim == 1:
                X_t = X_t.reshape(-1, 1)
            if X_t.shape[1] != cont_dim:
                raise ValueError(f"X has {X_t.shape[1]} columns but trained model expects {cont_dim}.")
            x_means = self._x_means.to(device=device, dtype=torch.float32)
            x_stds = self._x_stds.to(device=device, dtype=torch.float32)
            X_t = X_t.to(device=device)
            x_cont = _apply_nanstandardise(X_t, x_means, x_stds)

        # Prepare X_cat (validate dtype + range; bring to device).
        x_cat_t: torch.Tensor | None = None
        if X_cat is not None:
            x_cat_t, _ = _validate_and_normalise_cat(X_cat, cat_levels)
            x_cat_t = x_cat_t.to(device=device)

        with torch.no_grad():
            pi = net(x_cont, x_cat_t)
        return pi


# Class to store the results


def cash_posterior_means(
    X,
    betahat,
    sebetahat,
    n_epochs=20,
    n_layers=4,
    num_classes=20,
    hidden_dim=64,
    batch_size=128,
    lr=0.001,
    model_param=None,
    penalty=1.5,
    device: torch.device | None = None,
):
    """
    GPU-native CASH training and posterior computation.

    Fit a CASH (Covariate Adaptive Shrinkage) model and compute posterior means,
    second moments, and standard deviations.

    Parameters
    ----------
    X : torch.Tensor or np.ndarray
        Covariates for each observation, shape (n_samples, n_features).
    betahat : torch.Tensor or np.ndarray
        Observed effect estimates, shape (n_samples,).
    sebetahat : torch.Tensor or np.ndarray
        Standard errors of the effect estimates, shape (n_samples,).
    n_epochs : int, optional
        Number of training epochs (default=20).
    n_layers : int, optional
        Number of hidden layers in the neural network (default=4).
    num_classes : int, optional
        Number of mixture components (default=20).
    hidden_dim : int, optional
        Number of hidden units in each layer (default=64).
    batch_size : int, optional
        Batch size for training (default=128).
    lr : float, optional
        Learning rate for the optimizer (default=0.001).
    model_param : dict, optional
        Pre-trained model parameters to initialize the network.
    penalty : float, optional
        Penalty for spike probability (default=1.5).

    Returns
    -------
    cash_PosteriorMeanNorm
        Container with posterior means, standard deviations, and model parameters.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- to tensors on device
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    betahat = torch.as_tensor(betahat, dtype=torch.float32, device=device)
    sebetahat = torch.as_tensor(sebetahat, dtype=torch.float32, device=device)

    # ---- standardize on device
    X_scaled = standard_scale(X)  # your function returns just scaled tensor

    # ---- mixture scales (ensure tensor on device)
    scale = autoselect_scales_mix_norm(betahat=betahat, sebetahat=sebetahat, max_class=num_classes)
    if not isinstance(scale, torch.Tensor):
        scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
    else:
        scale = scale.to(device=device, dtype=torch.float32)

    # ---- dataset / loader (CUDA tensors => num_workers must be 0)
    dataset = DensityRegressionDataset(X_scaled, betahat, sebetahat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # ---- model / optimizer on device
    input_dim = X_scaled.shape[1]
    model_cash = CashNet(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, n_layers=n_layers).to(
        device
    )
    if model_param is not None:
        model_cash.load_state_dict(model_param)
    optimizer_cash = optim.Adam(model_cash.parameters(), lr=lr)

    # ---- training
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for inputs, targets, noise_std in dataloader:
            # Compute (log)likelihood for this batch and current global scales
            batch_loglik = get_data_loglik_normal_torch(
                betahat=targets, sebetahat=noise_std, location=0 * scale, scale=scale
            )
            optimizer_cash.zero_grad()
            outputs = model_cash(inputs)
            cash_loss = pen_loglik_loss(pred_pi=outputs, marginal_log_lik=batch_loglik, penalty=penalty)
            cash_loss.backward()
            optimizer_cash.step()
            epoch_loss += cash_loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"[CASH] Epoch {epoch + 1}/{n_epochs} | Loss: {epoch_loss / max(1, len(dataloader)):.4f}")

    # ---- full-batch inference (no grad)
    model_cash.eval()
    with torch.no_grad():
        train_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
        for X_batch, _, _ in train_loader:
            all_pi_values = model_cash(X_batch)  # (N, K)
        data_loglik = get_data_loglik_normal_torch(
            betahat=betahat, sebetahat=sebetahat, location=0 * scale, scale=scale
        )  # (N, K)

        # Allocate outputs on device
        J = betahat.shape[0]
        post_mean = torch.empty(J, dtype=torch.float32, device=device)
        post_mean2 = torch.empty(J, dtype=torch.float32, device=device)
        post_sd = torch.empty(J, dtype=torch.float32, device=device)

        # Per-observation posterior (kept as-is; can be vectorized later)
        eps = 1e-300
        for i in range(J):
            log_pi_i = torch.log(torch.clamp(all_pi_values[i, :], min=eps))
            res_i = posterior_mean_norm(
                betahat=betahat[i : i + 1],
                sebetahat=sebetahat[i : i + 1],
                log_pi=log_pi_i,
                data_loglik=data_loglik[i, :],
                location=[0],  # your routine expects this form
                scale=scale,
            )
            post_mean[i] = res_i.post_mean
            post_mean2[i] = res_i.post_mean2
            post_sd[i] = res_i.post_sd

        # ---- proper full negative marginal log-likelihood (no penalty).
        # Computed in log space via logsumexp (no `clamp(min=1e-10)` floor on
        # the inner density). This is what `cebmf.py:299`'s `kl_l[k] = (-loss)
        # - nm_ll_L` formula expects, and matches the `cebnm/emdn.py:288-298`
        # convention.
        log_pi_full = torch.log(all_pi_values.clamp_min(eps))  # (N, K)
        log_marginal_per_obs = torch.logsumexp(data_loglik + log_pi_full, dim=1)  # (N,)
        full_marginal_ll = float(log_marginal_per_obs.sum().item())

    return cash_PosteriorMeanNorm(
        post_mean=post_mean,
        post_mean2=post_mean2,
        post_sd=post_sd,
        pi_np=all_pi_values,  # (N, K) on device
        loss=-full_marginal_ll,
        scale=scale,  # (K,) on device
        model_param=model_cash.state_dict(),
    )
