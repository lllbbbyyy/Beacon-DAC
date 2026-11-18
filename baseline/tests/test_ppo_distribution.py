import itertools
import unittest

import torch

from baseline.ppo_distribution import HardwareDistribution


class HardwareProbabilityTests(unittest.TestCase):
    def test_normalization_and_entropy(self):
        dims = [2] + [2] * 6 + [2]
        logits = torch.linspace(-1, 1, sum(dims))[None, :]
        distribution = HardwareDistribution(dims, [1, 2]).proba_distribution(logits)
        actions = []
        for count_index, count in enumerate((1, 2)):
            for values in itertools.product(range(2), repeat=3 * count + 1):
                actions.append([count_index] + list(values[:-1]) + [0] * (6 - 3 * count) + [values[-1]])
        log_probs = distribution.log_prob(torch.tensor(actions))
        probabilities = log_probs.exp()
        torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))
        torch.testing.assert_close(distribution.entropy()[0], -(probabilities * log_probs).sum())

    def test_unused_slots_have_no_score_or_gradient(self):
        logits = torch.randn(1, 16, requires_grad=True)
        distribution = HardwareDistribution([2] * 8, [1, 2]).proba_distribution(logits)
        left = torch.zeros((1, 8), dtype=torch.long)
        right = left.clone()
        right[:, 4:7] = 1
        torch.testing.assert_close(distribution.log_prob(left), distribution.log_prob(right))
        distribution.log_prob(left).sum().backward()
        torch.testing.assert_close(logits.grad[:, 8:14], torch.zeros((1, 6)))
        for _ in range(20):
            action = distribution.sample()
            if action[0, 0] == 0:
                self.assertTrue(torch.all(action[:, 4:7] == 0))


if __name__ == "__main__":
    unittest.main()
