from __future__ import annotations

import torch

from dpl_rhf.rhfb.fock_checkpoint import exact_energy_block


def test_exact_energy_block_recomputes_backward_and_matches_plain_gradients():
    torch.manual_seed(81)
    x_plain = torch.randn(9, dtype=torch.float64, requires_grad=True)
    y_plain = torch.randn(9, dtype=torch.float64, requires_grad=True)
    x_check = x_plain.detach().clone().requires_grad_()
    y_check = y_plain.detach().clone().requires_grad_()
    probe = torch.randn((), dtype=torch.float64)
    plain_calls = {"count": 0}
    checked_calls = {"count": 0}

    def callback(counter):
        def evaluate(x, y):
            counter["count"] += 1
            mixed = torch.sin(x * y) + x.square() * torch.exp(-0.1 * y.square())
            return torch.sum(mixed.square()) * probe
        return evaluate

    expected = callback(plain_calls)(x_plain, y_plain)
    actual = exact_energy_block(callback(checked_calls), x_check, y_check)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(x_check.grad, x_plain.grad, atol=0.0, rtol=0.0)
    torch.testing.assert_close(y_check.grad, y_plain.grad, atol=0.0, rtol=0.0)
    assert plain_calls["count"] == 1
    assert checked_calls["count"] == 2


def test_exact_energy_block_skips_checkpoint_without_gradients():
    value = torch.tensor([1.0, 2.0], dtype=torch.float64)
    seen = {"count": 0}

    def evaluate(x):
        seen["count"] += 1
        return x.square().sum()

    result = exact_energy_block(evaluate, value)
    assert result.item() == 5.0
    assert seen["count"] == 1
