"""Mixture Density Network 

  features (F,)  ->  MLP trunk (SiLU)  ->  3 heads:
                       logits  (J,)  -> softmax    => weights pi_j (simplex)
                       mu      (J,)                => component means
                       log_sig (J,)  -> clamp+exp  => component stds > 0

Output: a J-component Gaussian mixture ~ conditional law  mu_T | (mu_0, W0)"""

import copy
import torch
import torch.nn as nn
import torch.distributions as D

#To be deleated
# def mlp(din: int, hidden: int, dout: int, depth: int = 2) -> nn.Sequential:
#     layers, d = [], din
#     for _ in range(depth):
#         layers += [nn.Linear(d, hidden), nn.SiLU()]   # smooth activation
#         d = hidden
#     layers.append(nn.Linear(d, dout))
#     return nn.Sequential(*layers)


class MDN(nn.Module):
    "d >= 1: each of the J components is N(mu_j, diag(sigma_j^2)) on R^d"
    def __init__(self, in_dim: int, hidden: int = 128, J: int = 5,
                 depth: int = 2, d: int = 1):
        super().__init__()
        self.J, self.d = J, d
        body, w = [], in_dim
        for _ in range(depth):
            body += [nn.Linear(w, hidden), nn.SiLU()]
            w = hidden
        self.body = nn.Sequential(*body)
        self.head_logits = nn.Linear(hidden, J)
        self.head_mu = nn.Linear(hidden, J * d)
        self.head_logsig = nn.Linear(hidden, J * d)

    def forward(self, f: torch.Tensor):
        """clamp(-5, 2) on log-sigma: prevents the "dead component" failure mode
        (sigma -> 0, loss -> -inf, exploding gradient)"""
        h = self.body(f)
        B = h.shape[0]
        logits = self.head_logits(h)                                 # (B, J)
        mu = self.head_mu(h).view(B, self.J, self.d)                 # (B, J, d)
        sigma = (self.head_logsig(h).clamp(-5.0, 2.0).exp()
                 .view(B, self.J, self.d))                           # safeguard #2
        return logits, mu, sigma

    def mixture(self, f: torch.Tensor) -> D.MixtureSameFamily:
        logits, mu, sigma = self(f)
        comp = D.Independent(D.Normal(mu, sigma), 1)
        return D.MixtureSameFamily(D.Categorical(logits=logits), comp)



def nll(model: MDN, feats: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """feats: (B, F), x: (B, K) -> scalar.  x.T => (K, B) against batch (B,)."""
    if x.dim() == 2:
        x = x.unsqueeze(-1)                       # (B, K, 1)
    return -model.mixture(feats).log_prob(x.transpose(0, 1)).mean()



def train_mdn(
    model: MDN,
    F_tr: torch.Tensor,
    X_tr: torch.Tensor,
    F_va: torch.Tensor,
    X_va: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    k_keep: int,
    seed: int = 0,
    log_every: int = 50,
    show: bool = True,
    eval_error_fn=None,
    eval_every: int = 10,
):
    """Train the MDN by conditional negative log-likelihood.

    eval_error_fn is an optional callback, typically computing validation W2.
    It is recorded for analysis but is not used for model selection.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=epochs,
        eta_min=lr * 1e-2,
    )

    generator = torch.Generator().manual_seed(seed)

    n = F_tr.shape[0]
    N = X_tr.shape[1]
    k_keep = min(k_keep, N)

    hist = {
        "train": [],
        "val": [],
        "lr": [],
        "error_epoch": [],
        "eval_error": [],
    }

    best_val = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(n, generator=generator)

        total_loss = 0.0
        total_count = 0

        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            features_batch = F_tr[idx]

            # For every scenario, randomly retain k_keep particles.
            particle_indices = torch.randint(
                low=0,
                high=N,
                size=(idx.shape[0], k_keep),
                generator=generator,
            )

            particles_batch = X_tr[idx]

            if particles_batch.dim() == 3:
                # X has shape (B, N, d).
                particle_indices = particle_indices.unsqueeze(-1).expand(
                    -1,
                    -1,
                    particles_batch.shape[-1],
                )

            selected_particles = torch.gather(
                particles_batch,
                dim=1,
                index=particle_indices,
            )

            loss = nll(model, features_batch, selected_particles)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * idx.shape[0]
            total_count += idx.shape[0]

        sched.step()

        # Validation NLL.
        model.eval()
        with torch.no_grad():
            validation_nll = nll(model, F_va, X_va).item()

        train_nll = total_loss / total_count

        hist["train"].append(train_nll)
        hist["val"].append(validation_nll)
        hist["lr"].append(float(opt.param_groups[0]["lr"]))

        # Optional distributional validation error.
        should_evaluate_error = (
            eval_error_fn is not None
            and (
                epoch == 1
                or epoch % eval_every == 0
                or epoch == epochs
            )
        )

        if should_evaluate_error:
            distribution_error = float(eval_error_fn(model))
            hist["error_epoch"].append(epoch)
            hist["eval_error"].append(distribution_error)

        # Model selection remains based on validation NLL.
        if validation_nll < best_val:
            best_val = validation_nll
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if show and (epoch % log_every == 0 or epoch == 1):
            extra_message = ""

            if hist["error_epoch"] and hist["error_epoch"][-1] == epoch:
                extra_message = (
                    f"  eval error {hist['eval_error'][-1]:.4f}"
                )

            print(
                f"    [MDN] epoch {epoch:4d}/{epochs}  "
                f"NLL train {train_nll:.4f}  "
                f"val {validation_nll:.4f}"
                f"{extra_message}"
            )

    hist["best_epoch"] = best_epoch

    if best_state is None:
        raise RuntimeError("No valid MDN checkpoint was produced.")

    model.load_state_dict(best_state)

    return hist, best_val