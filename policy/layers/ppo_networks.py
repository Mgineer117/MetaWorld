import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal, Categorical
from policy.layers.building_blocks import MLP


class PPO_Actor(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        a_dim: int,
        activation: nn.Module = nn.Tanh(),
    ):
        super(PPO_Actor, self).__init__()

        # |A| duplicate networks
        self.act = activation

        self.action_dim = a_dim

        self.model = MLP(
            input_dim, hidden_dim, a_dim, activation=self.act, initialization="actor"
        )

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ):
        logits = self.model(state)

        ### Shape the output as desired
        mu = logits
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)

        covariance_matrix = torch.diag_embed(std**2)  # Variance is std^2
        dist = MultivariateNormal(loc=mu, covariance_matrix=covariance_matrix)

        if deterministic:
            a = mu
        else:
            a = dist.rsample()

        logprobs = dist.log_prob(a).unsqueeze(-1)
        probs = torch.exp(logprobs)

        entropy = dist.entropy()

        return a, {
            "dist": dist,
            "probs": probs,
            "logprobs": logprobs,
            "entropy": entropy,
        }

    def log_prob(self, dist: torch.distributions, actions: torch.Tensor):
        """
        Actions must be tensor
        """
        actions = actions.squeeze() if actions.shape[-1] > 1 else actions
        logprobs = dist.log_prob(actions).unsqueeze(-1)
        return logprobs

    def entropy(self, dist: torch.distributions):
        """
        For code consistency
        """
        return dist.entropy().unsqueeze(-1)


class PPO_Critic(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self, input_dim: int, hidden_dim: list, activation: nn.Module = nn.Tanh()
    ):
        super(PPO_Critic, self).__init__()

        # |A| duplicate networks
        self.act = activation

        self.model = MLP(
            input_dim, hidden_dim, 1, activation=self.act, initialization="critic"
        )

    def forward(self, x: torch.Tensor):
        value = self.model(x)
        return value
