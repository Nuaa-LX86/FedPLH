import math
from collections import OrderedDict

import torch
import torch.nn.functional as F

from training.sota_adapters import (
    EvidentialDiceLoss3D,
    FedCLAMDiceFIMLoss3D,
    aggregate_fedclam_states,
    calculate_overfitting_penalty,
    calculate_vlr,
    fedevi_adjust_weights,
)


def test_fedevi_loss_is_finite_and_differentiable():
    logits = torch.randn(2, 4, 4, 4, 4, requires_grad=True)
    target = torch.randint(0, 4, (2, 4, 4, 4))
    loss = EvidentialDiceLoss3D()(logits, target, round_number=3)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_fedevi_dice_reduction_matches_official_batch_formula():
    logits = torch.tensor(
        [
            [[[[0.0, 0.5]]], [[[0.2, -0.3]]]],
            [[[[1.0, -0.5]]], [[[-0.4, 0.7]]]],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor([[[[0, 1]]], [[[1, 0]]]])
    criterion = EvidentialDiceLoss3D(kl_weight=0.0)
    actual = criterion(logits, target, round_number=0)

    alpha = torch.exp(logits) + 1.0
    probabilities = alpha / alpha.sum(dim=1, keepdim=True)
    one_hot = F.one_hot(target, 2).movedim(-1, 1).float()
    expected_dice = 0.0
    for class_index in range(2):
        pred = probabilities[:, class_index]
        truth = one_hot[:, class_index]
        intersection = (pred * truth).sum()
        denominator = pred.square().sum() + truth.sum()
        expected_dice += (2.0 * intersection + 1e-5) / (denominator + 1e-5)
    expected = 1.0 - expected_dice / 2.0
    assert torch.allclose(actual, expected)


def test_fedevi_weight_formula_matches_direct_reference():
    base = [0.25, 0.75]
    scores = [(0.2, 0.4), (0.1, 0.5)]
    actual = fedevi_adjust_weights(base, scores)
    raw = [0.25 + 0.2 / 0.4, 0.75 + 0.1 / 0.5]
    expected = [value / sum(raw) for value in raw]
    assert torch.allclose(torch.tensor(actual), torch.tensor(expected))


def test_fedclam_scalar_controls_match_official_equations():
    initial, trained, train = 1.2, 0.8, 0.6
    expected_vlr = 1.0 / (1.0 + math.exp(-(initial - trained) / (trained + 1e-6)))
    expected_penalty = 1.0 - min(train / (trained + 1e-6), 1.0)
    assert abs(calculate_vlr(initial, trained, 1.0) - expected_vlr) < 1e-12
    assert abs(calculate_overfitting_penalty(train, trained, 1.0) - expected_penalty) < 1e-12


def test_fedclam_round_zero_matches_equal_fedavg():
    base = OrderedDict(weight=torch.tensor([0.0]), counter=torch.tensor(1))
    states = [
        OrderedDict(weight=torch.tensor([1.0]), counter=torch.tensor(1)),
        OrderedDict(weight=torch.tensor([3.0]), counter=torch.tensor(1)),
    ]
    metrics = [
        {"val_loss_ratio": 0.5, "overfitting_penalty": 0.25},
        {"val_loss_ratio": 0.5, "overfitting_penalty": 0.25},
    ]
    result, _ = aggregate_fedclam_states(
        base, states, [0, 1], metrics, {}, round_index=0
    )
    assert torch.equal(result["weight"], torch.tensor([2.0]))
    assert torch.equal(result["counter"], torch.tensor(1))


def test_fedclam_3d_loss_warmup_and_active_paths():
    logits = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
    target = torch.randint(0, 4, (1, 8, 8, 8))
    image = torch.randn(1, 4, 8, 8, 8)
    criterion = FedCLAMDiceFIMLoss3D(4, fim_warmup_rounds=1, pooled_size=4)
    warmup = criterion(logits, target, image, round_index=0)
    active = criterion(logits, target, image, round_index=1)
    assert torch.isfinite(warmup)
    assert torch.isfinite(active)
    expected = 0.5 * criterion.dice(logits, target) + 0.5 * criterion.fim(
        logits, target, image
    )
    assert torch.allclose(active, expected)
