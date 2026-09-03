from __future__ import annotations

from typing import Dict, Mapping, Sequence


class FedMPQProxyScheduler:
    """
    Clean-room implementation of FedMPQ Algorithm 2.

    The server aggregates local bit assignments and ranks layers by
    parameter_count * (previous_local_bit_reduction + 1). It prunes from the
    front of that order and grows from the back until each client's weighted
    average bit-width budget is met.
    """

    def __init__(
        self,
        layer_parameter_counts: Mapping[str, int],
        client_budgets: Sequence[int],
    ):
        if not layer_parameter_counts:
            raise ValueError("At least one quantized layer is required")
        if not client_budgets:
            raise ValueError("At least one client budget is required")

        self.layer_parameter_counts = {
            str(name): int(count)
            for name, count in layer_parameter_counts.items()
        }
        if any(count <= 0 for count in self.layer_parameter_counts.values()):
            raise ValueError("Layer parameter counts must be positive")

        self.client_budgets = [int(value) for value in client_budgets]
        if any(value not in {2, 4, 6, 8} for value in self.client_budgets):
            raise ValueError("Client budgets must be in {2, 4, 6, 8}")

        self.client_policies: Dict[int, Dict[str, int]] = {
            client_id: {
                layer_name: budget
                for layer_name in self.layer_parameter_counts
            }
            for client_id, budget in enumerate(self.client_budgets)
        }
        self.last_delta: Dict[int, Dict[str, int]] = {
            client_id: {
                layer_name: 0
                for layer_name in self.layer_parameter_counts
            }
            for client_id in range(len(self.client_budgets))
        }

    def policy_for(self, client_id: int) -> Dict[str, str]:
        return {
            name: f"INT{bits}"
            for name, bits in self.client_policies[int(client_id)].items()
        }

    def integer_policy_for(self, client_id: int) -> Dict[str, int]:
        return dict(self.client_policies[int(client_id)])

    def average_bits(self, policy: Mapping[str, int]) -> float:
        total = float(sum(self.layer_parameter_counts.values()))
        return float(
            sum(
                self.layer_parameter_counts[name] * int(policy[name])
                for name in self.layer_parameter_counts
            )
            / total
        )

    def update(
        self,
        participating_clients: Sequence[int],
        local_bit_assignments: Sequence[Mapping[str, int]],
        local_bit_deltas: Sequence[Mapping[str, int]],
        aggregation_weights: Sequence[float],
    ) -> None:
        if not participating_clients:
            return
        if not (
            len(participating_clients)
            == len(local_bit_assignments)
            == len(local_bit_deltas)
            == len(aggregation_weights)
        ):
            raise ValueError("FedMPQ update inputs must have equal length")

        aggregate_bits = {
            name: sum(
                float(weight) * int(assignment[name])
                for assignment, weight in zip(
                    local_bit_assignments,
                    aggregation_weights,
                )
            )
            for name in self.layer_parameter_counts
        }
        rounded_base = {
            name: max(1, min(8, int(round(value))))
            for name, value in aggregate_bits.items()
        }

        for client_id, delta in zip(
            participating_clients,
            local_bit_deltas,
        ):
            self.last_delta[int(client_id)] = {
                name: int(delta[name])
                for name in self.layer_parameter_counts
            }

        for client_id, budget in enumerate(self.client_budgets):
            self.client_policies[client_id] = self._fit_budget(
                rounded_base,
                self.last_delta[client_id],
                budget,
            )

    def _fit_budget(
        self,
        base_policy: Mapping[str, int],
        bit_delta: Mapping[str, int],
        budget: int,
    ) -> Dict[str, int]:
        policy = {name: int(bits) for name, bits in base_policy.items()}
        order = sorted(
            policy,
            key=lambda name: (
                -self.layer_parameter_counts[name]
                * (int(bit_delta.get(name, 0)) + 1),
                name,
            ),
        )

        cursor = 0
        while (
            self.average_bits(policy) > float(budget) + 1e-12
            and cursor < len(order)
        ):
            name = order[cursor]
            if policy[name] > 1:
                policy[name] -= 1
            else:
                cursor += 1

        cursor = len(order) - 1
        while (
            self.average_bits(policy) < float(budget) - 1e-12
            and cursor >= 0
        ):
            name = order[cursor]
            if policy[name] < 8:
                candidate = dict(policy)
                candidate[name] += 1
                if (
                    self.average_bits(candidate)
                    <= float(budget) + 1e-12
                ):
                    policy = candidate
                else:
                    cursor -= 1
            else:
                cursor -= 1

        return policy
