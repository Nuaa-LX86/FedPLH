import copy

import torch

from models.fedmpq_proxy import FedMPQProxy
from models.unet3d import UNet3D
from training.fedmpq_proxy import FedMPQProxyScheduler


def test_fedmpq_proxy_forward_and_state_roundtrip():
    model = FedMPQProxy(
        UNet3D(n_channels=4, n_classes=4, base_filters=2),
        default_bits=8,
    )
    policy = {name: "INT4" for name in model.layer_bits}
    model.update_policy(policy)

    output = model(torch.randn(1, 4, 16, 16, 16))
    assert output.shape == (1, 4, 16, 16, 16)
    assert abs(model.average_bit_width() - 4.0) < 1e-12
    assert model.profile_precision() == "INT4"
    assert set(model.quantization_feedback()) == set(model.layer_bits)
    regularizer = model.bit_group_lasso()
    assert torch.isfinite(regularizer)
    regularizer.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
    )
    pruned = model.prune_msb(0.03)
    assert set(pruned) == set(model.layer_bits)

    clone = copy.deepcopy(model)
    clone.load_state_dict(model.state_dict(), strict=True)
    clone.update_policy("FP32")
    output_fp32 = clone(torch.randn(1, 4, 16, 16, 16))
    assert output_fp32.shape == (1, 4, 16, 16, 16)

    one_bit_policy = {name: "INT1" for name in model.layer_bits}
    model.update_policy(one_bit_policy)
    one_bit_output = model(torch.randn(1, 4, 16, 16, 16))
    assert torch.isfinite(one_bit_output).all()


def test_fedmpq_proxy_scheduler_respects_client_budgets():
    scheduler = FedMPQProxyScheduler(
        {"large": 100, "small": 10},
        [2, 4, 6, 8],
    )
    local_bits = [
        {"large": 4, "small": 3},
        {"large": 8, "small": 7},
    ]
    deltas = [
        {"large": 0, "small": 1},
        {"large": 0, "small": 1},
    ]
    scheduler.update([1, 3], local_bits, deltas, [0.4, 0.6])

    for client_id, budget in enumerate([2, 4, 6, 8]):
        policy = {
            name: int(value[3:])
            for name, value in scheduler.policy_for(client_id).items()
        }
        assert all(1 <= value <= 8 for value in policy.values())
        assert abs(scheduler.average_bits(policy) - budget) <= 1.0
