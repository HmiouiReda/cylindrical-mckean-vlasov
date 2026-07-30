"""Pham-Warin cylindrical network

  V_hat(x, mu_0, w) = Psi( x , < phi , N_hat(features) > )

The inner functions phi = {sin(k.), cos(k.)} belong to the SAME Fourier
family as the features, so the integral against the predicted Gaussian
mixture is ANALYTIC (characteristic functions):

  E_nu[sin(kY)] = sum_j pi_j exp(-k^2 sigma_j^2 / 2) sin(k mu_j)  (same, cos)"""

"""
two input variants for Psi, trained on IDENTICAL (x, y) draws:
  * ORACLE integrals  <phi, mu_T>      -- closed form (OU) 
  * MDN integrals     <phi, N_hat(.)>
  * EMPIRICAL integrals <phi, mu_hat_N> -- the nested / plug-in estimator
"""

import copy
import numpy as np
import torch
import torch.nn as nn

from mdn import MDN




# Fourier encodings of the MDN mixture and empirical particle measure
def mixture_fourier(logits: torch.Tensor, mu: torch.Tensor,
                    sigma: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Closed form against the predicted diagonal-Gaussian mixture, any d.
    logits: (B, J);  mu, sigma: (B, J) [legacy 1-d] or (B, J, d);
    freqs: (P,) [1-d] or (P, d)  ->  (B, 2P) = [sin | cos] with
    E[sin(<k, Y>)] = sum_j pi_j exp(-1/2 sum_i k_i^2 sigma_ji^2)
                     sin(<k, mu_j>)."""
    if mu.dim() == 2:
        mu, sigma = mu.unsqueeze(-1), sigma.unsqueeze(-1)    # (B, J, 1)
    fr = freqs if freqs.dim() == 2 else freqs.view(-1, 1)    # (P, d)
    pi = torch.softmax(logits, dim=-1)                       # (B, J)
    k2 = (fr ** 2)[None, None]                               # (1, 1, P, d)
    damp = torch.exp(-0.5 * (k2 * sigma.unsqueeze(2) ** 2).sum(-1))  # (B,J,P)
    arg = (fr[None, None] * mu.unsqueeze(2)).sum(-1)         # (B, J, P)
    s = (pi.unsqueeze(-1) * damp * torch.sin(arg)).sum(dim=1)
    c = (pi.unsqueeze(-1) * damp * torch.cos(arg)).sum(dim=1)
    return torch.cat([s, c], dim=-1)                         # (B, 2P)

def mdn_integrals(mdn: MDN, feats_std: torch.Tensor,
                  freqs: torch.Tensor) -> torch.Tensor:
    """<phi, N_hat(features)> with the MDN FROZEN."""
    with torch.no_grad():
        logits, mu, sigma = mdn(feats_std)
        return mixture_fourier(logits, mu, sigma, freqs)


def empirical_fourier(X: np.ndarray, freqs: np.ndarray) -> torch.Tensor:
    """<phi, mu_hat_N>: plain sample means of sin(<k, X>), cos(<k, X>) over
    the particles of each scenario -- exactly what a nested / plug-in
    estimator has at evaluation time.  X: (M, N[, d]) -> (M, 2P)."""
    X_ = X[..., None] if X.ndim == 2 else X                  # (M, N, d)
    fr = np.asarray(freqs)
    fr = fr.reshape(-1, X_.shape[-1]) if fr.ndim == 2 else fr.reshape(-1, 1)
    arg = np.einsum("mnd,pd->mnp", X_, fr)                   # (M, N, P)
    s = np.sin(arg).mean(axis=1)
    c = np.cos(arg).mean(axis=1)
    return torch.as_tensor(np.concatenate([s, c], axis=1), dtype=torch.float32)



# Benchmark target functional
def V_target(x: np.ndarray, m1: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Moment functional, d = 1:  V = x m1 + Var + sin(x).  Depends on mu
    only through its first two moments -- kept as the EASY functional."""
    return x * m1 + var + np.sin(x)


def V_target_nd(x: np.ndarray, m1: np.ndarray, tr: np.ndarray) -> np.ndarray:
    """Moment functional, d >= 2:  V = <x, m1> + tr Cov + sin(x_1).
    x: (..., d), m1 broadcastable to x, tr broadcastable to the result."""
    return (x * m1).sum(-1) + tr + np.sin(x[..., 0])


def V_target_quantile(x: np.ndarray, iqr: np.ndarray,
                      cdf0: np.ndarray) -> np.ndarray:
    """HARD functional (d = 1):  V = x * IQR(mu) + F_mu(0) + sin(x), with
    IQR = q_0.75 - q_0.25.  Neither the interquartile range nor the mass
    P(X <= 0) is a function of finitely many moments: for the bimodal
    double well, F_mu(0) is precisely the left-well occupancy.  This
    stresses the cylindrical layer far beyond the moment functional."""
    return x * iqr + cdf0 + np.sin(x)



def mlp(din: int, hidden: int, dout: int, depth: int = 2) -> nn.Sequential:
    layers, d = [], din
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU()]   # smooth activation
        d = hidden
    layers.append(nn.Linear(d, dout))
    return nn.Sequential(*layers)


class Psi(nn.Module):
    "Input: [x (d) | integrals of a measure representation (2P)]."
    def __init__(self, n_integrals: int, hidden: int = 128, depth: int = 2,
                 x_dim: int = 1):
        super().__init__()
        self.net = mlp(x_dim + n_integrals, hidden, 1, depth)

    def forward(self, z):
        return self.net(z)



# (x, y) draws + attachment of any integral variant
def psi_xy(XT: np.ndarray, f1_ref: np.ndarray, f2_ref: np.ndarray, k_x: int,
           rng: np.random.Generator, functional: str = "moments", 
           empirical_targets: bool = False):
    """Draw the evaluation points x ~ mu_T and the targets y ONCE, so that
    all integral variants are trained/evaluated on identical data.

    functional: "moments" -> (f1, f2) = (mean, variance/trace);
                "quantile" -> (f1, f2) = (IQR, F_mu(0)), d = 1 only.

    Returns x_t (rows, d), y_t (rows, 1), k_x  with rows = M * k_x, scenario-major order."""
    M, N = XT.shape[0], XT.shape[1]
    d = XT.shape[2] if XT.ndim == 3 else 1
    if k_x <= 0 or k_x >= N:
        x = XT.copy()
        k_x = N
    else:
        cols = rng.integers(0, N, size=(M, k_x))
        x = XT[np.arange(M)[:, None], cols]                  # (M, k_x[, d])

    if functional == "moments":
        if empirical_targets:
            if d == 1:
                f1, f2 = XT.mean(axis=1), XT.var(axis=1)
            else:
                f1, f2 = XT.mean(axis=1), XT.var(axis=1).sum(-1)   # trace
        else:
            f1, f2 = f1_ref, f2_ref
        if d == 1:
            y = V_target(x, f1[:, None], f2[:, None])        # (M, k_x)
        else:
            y = V_target_nd(x, f1[:, None, :], f2[:, None])
    elif functional == "quantile":
        assert d == 1, "the quantile functional is defined for d = 1"
        if empirical_targets:
            q = np.quantile(XT, [0.25, 0.75], axis=1)
            f1, f2 = q[1] - q[0], (XT <= 0.0).mean(axis=1)
        else:
            f1, f2 = f1_ref, f2_ref
        y = V_target_quantile(x, f1[:, None], f2[:, None])
    else:
        raise ValueError(f"unknown functional '{functional}'")

    x_t = torch.as_tensor(x.reshape(-1, d), dtype=torch.float32)
    y_t = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32)
    return x_t, y_t, k_x


def attach_integrals(x_t: torch.Tensor, ints: torch.Tensor,
                     k_x: int) -> torch.Tensor:
    """[x | integrals], repeating the per-scenario integrals over the k_x
    evaluation points (scenario-major order, matching psi_xy)."""
    return torch.cat([x_t, torch.repeat_interleave(ints, k_x, dim=0)], dim=1)


def make_psi_inputs(ints: torch.Tensor, XT: np.ndarray, f1_ref: np.ndarray,
                    f2_ref: np.ndarray, k_x: int, rng: np.random.Generator,
                    functional: str = "moments", empirical_targets: bool = False):
    """Design + targets for ONE integral variant: draw (x, y) via psi_xy and
    attach the given per-scenario integrals.  Returns (Xin, y)."""
    x_t, y_t, k_eff = psi_xy(XT, f1_ref, f2_ref, k_x, rng, functional, empirical_targets)
    return attach_integrals(x_t, ints, k_eff), y_t


def psi_dataset(mdn: MDN, feats_std: torch.Tensor, XT: np.ndarray,
                f1_ref: np.ndarray, f2_ref: np.ndarray, k_x: int,
                freqs: torch.Tensor, rng: np.random.Generator,
                empirical_targets: bool = False, functional: str = "moments"):
    """The pipeline's own variant: closed-form integrals against the FROZEN
    MDN mixture, then the shared design assembly.  Returns (Xin, y)."""
    ints = mdn_integrals(mdn, feats_std, freqs)
    return make_psi_inputs(ints, XT, f1_ref, f2_ref, k_x, rng,functional, empirical_targets)



# MSE training loop
def train_regressor(model: nn.Module,
                    X_tr: torch.Tensor, y_tr: torch.Tensor,
                    X_va: torch.Tensor, y_va: torch.Tensor,
                    epochs: int, batch_size: int, lr: float,
                    seed: int = 0, log_every: int = 50, show: bool = True):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=lr * 1e-2)
    crit = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    n = X_tr.shape[0]
    hist = {"train": [], "val": []}
    best_val, best_state = float("inf"), None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot, cnt = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            pred = model(X_tr[idx])                   # 1. forward
            loss = crit(pred, y_tr[idx])              # 2. loss
            opt.zero_grad()                           # 3. reset grads
            loss.backward()                           # 4. backprop
            opt.step()                                # 5. Adam update
            tot += loss.item() * idx.shape[0]
            cnt += idx.shape[0]
        sched.step()

        model.eval()
        with torch.no_grad():
            va = crit(model(X_va), y_va).item()
        hist["train"].append(tot / cnt)
        hist["val"].append(va)
        if va < best_val:
            best_val = va
            best_state = copy.deepcopy(model.state_dict())
        if show and (ep % log_every == 0 or ep == 1):
            print(f"    Psi epoch {ep:4d}/{epochs}  "
                  f"MSE train {tot/cnt:.5f}  val {va:.5f}")

    model.load_state_dict(best_state)
    return hist, best_val
