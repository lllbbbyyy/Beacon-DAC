"""Probability over physical designs, marginalizing unused chiplet slots."""
import torch
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class HardwareDistribution(MultiCategoricalDistribution):
    def __init__(self, action_dims, chiplet_counts):
        super().__init__(action_dims)
        self.chiplet_counts = tuple(chiplet_counts)
        self.max_count = max(self.chiplet_counts)
        if len(action_dims) < 1 + 3 * self.max_count or action_dims[0] != len(chiplet_counts):
            raise ValueError("Distribution does not match the hardware codec")

    def _active(self, actions):
        counts = torch.as_tensor(self.chiplet_counts, device=actions.device)[actions[:, 0].long()]
        slots = torch.arange(self.max_count, device=actions.device)[None, :]
        return (slots < counts[:, None]).repeat_interleave(3, dim=1)

    def _canonical(self, actions):
        end = 1 + 3 * self.max_count
        actions[:, 1:end] *= self._active(actions)
        return actions

    def sample(self):
        return self._canonical(super().sample())

    def mode(self):
        return self._canonical(super().mode())

    def log_prob(self, actions):
        terms = torch.stack([dist.log_prob(action) for dist, action in
                             zip(self.distribution, actions.unbind(dim=1))], dim=1)
        end = 1 + 3 * self.max_count
        return (terms[:, 0] + (terms[:, 1:end] * self._active(actions)).sum(dim=1)
                + terms[:, end:].sum(dim=1))

    def entropy(self):
        # H(count) + E_count[H(active slots | count)] + H(system).
        terms = torch.stack([dist.entropy() for dist in self.distribution], dim=1)
        counts = torch.as_tensor(self.chiplet_counts, device=terms.device)
        active = (torch.arange(self.max_count, device=terms.device)[None, :] < counts[:, None])
        probability = self.distribution[0].probs @ active.to(terms.dtype)
        end = 1 + 3 * self.max_count
        return (terms[:, 0] + (terms[:, 1:end] * probability.repeat_interleave(3, dim=1)).sum(dim=1)
                + terms[:, end:].sum(dim=1))


class HardwarePolicy(ActorCriticPolicy):
    def __init__(self, *args, chiplet_counts, **kwargs):
        self.chiplet_counts = tuple(chiplet_counts)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_dist = HardwareDistribution(list(self.action_space.nvec), self.chiplet_counts)

    def _get_constructor_parameters(self):
        parameters = super()._get_constructor_parameters()
        parameters["chiplet_counts"] = self.chiplet_counts
        return parameters
