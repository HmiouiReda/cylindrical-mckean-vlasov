"""MDN: shapes, safeguards and a fast gradient-flow smoke test."""
import torch

from mdn import MDN, nll, train_mdn


def test_forward_shapes_and_safeguards():
    torch.manual_seed(0)
    m = MDN(in_dim=14, hidden=32, J=5)
    f = torch.randn(7, 14)
    logits, mu, sigma = m(f)
    assert logits.shape == (7, 5)
    assert mu.shape == sigma.shape == (7, 5, 1)      # (B, J, d), d = 1
    m2 = MDN(in_dim=14, hidden=32, J=4, d=2)
    lg2, mu2, sg2 = m2(f)
    assert lg2.shape == (7, 4) and mu2.shape == sg2.shape == (7, 4, 2)
    assert (sigma > 0).all()
    # clamp(-5, 2) on log-sigma
    assert (sigma >= torch.exp(torch.tensor(-5.0)) - 1e-6).all()
    assert (sigma <= torch.exp(torch.tensor(2.0)) + 1e-6).all()


def test_nll_finite():
    torch.manual_seed(0)
    m = MDN(in_dim=6, hidden=16, J=3)
    loss = nll(m, torch.randn(5, 6), torch.randn(5, 8))
    assert torch.isfinite(loss)


def test_training_reduces_nll():
    """Tiny deliberate-overfit check: a frozen set of targets must be
    memorisable, otherwise the gradient flow is broken."""
    torch.manual_seed(0)
    F = torch.randn(30, 6)
    X = torch.randn(30, 4)                     # FROZEN particles
    m = MDN(in_dim=6, hidden=32, J=3)
    hist, _ = train_mdn(m, F, X, F, X, epochs=600, batch_size=30,
                        lr=3e-3, k_keep=4, seed=0, show=False)
    assert hist["train"][-1] < hist["train"][0] - 0.5   # check if the error has decreased by at least 0.5
