"""Behavioral gates for causal, masked, matched learning recipes."""

import numpy as np
import pytest
import torch

from scripts.stage2_calibration import finite_population
from scripts.stage2_train import read_probabilities
from systems.stage2_learning import ARMS, build_model, objective, state_hash


@pytest.fixture(autouse=True)
def require_deterministic_operations():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize("arm", ["mean_ce", "gru_ce", "transformer_ce"])
def test_future_inputs_cannot_change_past_outputs(arm: str) -> None:
    torch.manual_seed(8)
    net = build_model(arm, 8).eval()
    x = torch.randn(2, 9, 8)
    altered = x.clone()
    altered[:, 5:] += 500
    with torch.no_grad():
        full = net(x)[:, :5]
        torch.testing.assert_close(full, net(altered)[:, :5], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(full, net(x[:, :5]), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("arm", ARMS)
def test_padded_readouts_do_not_affect_loss_or_gradients(arm: str) -> None:
    torch.manual_seed(3)
    logits = torch.randn(2, 6, 5, requires_grad=True)
    labels = torch.tensor([0, 4])
    weights = torch.tensor([[0.25, 0.5, 0.25, 0, 0, 0], [0.125, 0.25, 0.25, 0.25, 0.125, 0]])
    cw = torch.ones(5)
    reference = objective(logits, labels, weights, cw, arm)
    changed = logits.detach().clone()
    changed[weights == 0] = torch.tensor([-100.0, 100.0, 0.0, 0.0, 0.0])
    torch.testing.assert_close(reference, objective(changed, labels, weights, cw, arm))
    reference.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[weights == 0] == 0).all()


def test_matched_gru_initialization_and_suffix_upper_envelope() -> None:
    hashes = []
    for arm in ["gru_ce", "gru_time", "gru_suffix"]:
        torch.manual_seed(7)
        hashes.append(state_hash(build_model(arm, 8)))
    assert len(set(hashes)) == 1
    x = torch.randn(2, 5, 5)
    weights = torch.ones(2, 5) / 5
    labels = torch.tensor([1, 2])
    assert objective(x, labels, weights, torch.ones(5), "gru_suffix") >= objective(
        x, labels, weights, torch.ones(5), "gru_ce"
    )


def test_rare_event_truth_is_nonzero_even_when_no_error_is_observed() -> None:
    table, probabilities = finite_population(rare=True)
    mean = np.einsum("ckd,k->cd", table, probabilities).mean(0)
    assert np.all(table[:, 0] == 0)
    assert mean[0] == pytest.approx(0.002 * 8.75)
    assert mean[6] == pytest.approx(0.002 * 10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual deployment GPU")
@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("ema", [None, 0.05])
def test_cuda_actual_readout_path_is_causal(arm: str, ema: float | None) -> None:
    torch.manual_seed(8)
    net = build_model(arm, 512).cuda().eval()
    x = torch.randn(3, 57, 512, device="cuda")
    x[1, 30:] = 0
    index = torch.arange(20, device="cuda")[None].expand(3, -1)
    valid = torch.ones_like(index, dtype=torch.bool)
    original = read_probabilities(net, x, index, valid, ema=ema)
    changed = x.clone()
    changed[:, 20:] = torch.randn_like(changed[:, 20:]) * 100
    np.testing.assert_allclose(original, read_probabilities(net, changed, index, valid, ema=ema), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(
        original, read_probabilities(net, x[:, :20], index, valid, ema=ema), atol=1e-5, rtol=1e-5
    )
