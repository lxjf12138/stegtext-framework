from __future__ import annotations

import unittest
from decimal import Decimal, getcontext

import torch

from stegtext.v2.coders.range_ac import RangeAC


getcontext().prec = 150


def official_consistency_actions(probability_steps, secret_bits: str, width: int = 8) -> list[int]:
    """Independent Decimal fixture for the historical Consistency AC rule."""
    secret = Decimal(int(secret_bits, 2))
    lower, upper = Decimal(0), Decimal(2**width)
    actions: list[int] = []
    for probabilities in probability_steps:
        values = [Decimal(str(float(value))) for value in probabilities]
        total = sum(values)
        values = [value / total for value in values]
        span = upper - lower
        cursor = lower
        for index, probability in enumerate(values):
            next_cursor = cursor + probability * span
            if cursor <= secret <= next_cursor:
                actions.append(index)
                lower, upper = cursor, next_cursor
                break
            cursor = next_cursor
        else:
            raise AssertionError("fixture secret fell outside every arithmetic interval")
    return actions


class OfficialArithmeticFixtureTests(unittest.TestCase):
    def test_range_ac_matches_consistency_action_fixture(self) -> None:
        probabilities = [
            torch.tensor([0.25, 0.75], dtype=torch.float64),
            torch.tensor([0.60, 0.40], dtype=torch.float64),
            torch.tensor([0.20, 0.30, 0.50], dtype=torch.float64),
        ]
        secret_bits = "01011010"
        expected_actions = official_consistency_actions(probabilities, secret_bits)
        self.assertEqual(expected_actions, [1, 0, 1])

        encoder = RangeAC(precision=8).start("encode")
        stream = secret_bits + "0" * 64
        emitted: list[str] = []
        actions: list[int] = []
        for distribution in probabilities:
            bits, action = encoder.encode(distribution, stream, rng=None)
            emitted.append(bits)
            actions.append(action)
            stream = stream[len(bits) :]
        self.assertEqual(actions, expected_actions)

        decoder = RangeAC(precision=8).start("decode")
        decoded = "".join(
            decoder.decode(distribution, action, rng=None)
            for distribution, action in zip(probabilities, actions)
        )
        self.assertEqual(decoded, "".join(emitted))


if __name__ == "__main__":
    unittest.main()
