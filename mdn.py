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



def train_mdn(model: MDN,
              F_tr: torch.Tensor, X_tr: torch.Tensor,
              F_va: torch.Tensor, X_va: torch.Tensor,
              epochs: int, batch_size: int, lr: float,
              k_keep: int, seed: int = 0, log_every: int = 50,
              show: bool = True):
    """Canonical PyTorch loop: forward -> loss -> zero_grad -> backward ->
    step."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=lr * 1e-2)
    g = torch.Generator().manual_seed(seed)
    n, N = F_tr.shape[0], X_tr.shape[1]   # X_tr: (n, N[, d])
    k_keep = min(k_keep, N)
    hist = {"train": [], "val": []}
    best_val, best_state = float("inf"), None

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot, cnt = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            fb = F_tr[idx]
            # subsample K_keep particle columns per scenario (variance only)
            cols = torch.randint(0, N, (idx.shape[0], k_keep), generator=g)
            Xi = X_tr[idx]
            if Xi.dim() == 3:                     # (B, N, d)
                cols = cols.unsqueeze(-1).expand(-1, -1, Xi.shape[-1])
            xb = torch.gather(Xi, 1, cols)
            loss = nll(model, fb, xb)                 # 1. forward + 2. loss
            opt.zero_grad()                           # 3. reset grads
            loss.backward()                           # 4. backprop
            opt.step()                                # 5. Adam update
            tot += loss.item() * idx.shape[0]
            cnt += idx.shape[0]
        sched.step()

        model.eval()
        with torch.no_grad():                          # never build graphs in eval
            va = nll(model, F_va, X_va).item()
        tr = tot / cnt
        hist["train"].append(tr)
        hist["val"].append(va)
        if va < best_val:
            best_val = va
            best_state = copy.deepcopy(model.state_dict())
        if show and (ep % log_every == 0 or ep == 1):
            print(f"    [MDN] epoch {ep:4d}/{epochs}  "
                  f"NLL train {tr:.4f}  val {va:.4f}")

    model.load_state_dict(best_state)
    return hist, best_val
