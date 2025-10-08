"""rl_agent.py
----------------

On-the-fly reinforcement learning agent for the locomotion project.

This module defines a small actor–critic agent that can adapt a
pretrained feedforward policy using reward signals returned from a
Unity simulation.  It is designed to integrate with the existing
communication loop without major changes to your pipeline.

Key Features
============
- **Actor network** outputs mean actions for each degree of freedom.
  The actions are sampled from a Normal distribution with a fixed
  learnable standard deviation.  By default, the actor is
  initialised using the weights from an offline‐trained MLP so that
  on‑line training starts from a sensible walking policy.
- **Critic network** estimates the value of a state.  The critic
  shares the same hidden architecture as the actor but has a single
  scalar output.  It provides a baseline to reduce variance in
  policy gradient updates.
- **Per‑step updates** use an Advantage Actor–Critic (A2C) style
  algorithm: after receiving a reward and new observation, the agent
  computes the temporal difference error and performs a gradient
  update on both actor and critic.  This keeps the agent responsive
  to changes in the environment.
- **Gamma and learning rate** can be tuned via constructor
  arguments.  If you wish to perform more stable updates, you can
  accumulate transitions and call ``update()`` less frequently.

The agent exposes two main methods:

* :func:`select_action` – Given a raw observation array, returns a
  sampled action vector along with the log probability of that
  sample and the critic's value estimate.  You should call this to
  choose actions to send to Unity.
* :func:`update` – Performs a gradient update using the reward
  received for the previous action and the value estimate of the
  current state.  This should be called once per time step after
  receiving feedback from Unity.

Example usage::

    agent = RLAgent(obs_dim=24, action_dim=12, hidden_sizes=[128, 64], lr=1e-4)
    # Optionally load actor weights from a pretrained model
    agent.load_from_model(pretrained_model)

    obs = np.random.randn(24)
    action, logp, value = agent.select_action(obs)
    # send action to Unity, receive reward and next_obs
    agent.update(reward, next_obs, done=False)

See ``main.py`` for integration with the Unity communication loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class RLAgent:
    """Advantage Actor–Critic agent for continuous actions.

    This agent maintains separate actor and critic networks.  The
    actor outputs the mean action for each degree of freedom, and
    actions are sampled from a Normal distribution with a learnable
    log‑standard deviation.  The critic predicts a scalar value for
    each observation to serve as a baseline in the policy gradient.

    Parameters
    ----------
    obs_dim : int
        Dimension of the observation vector (input to actor/critic).
    action_dim : int
        Dimension of the action vector (number of DoF).  Each action
        represents a continuous target command in the range [-1, 1].
    hidden_sizes : list[int]
        Sizes of hidden layers for both actor and critic.  Defaults to
        two layers [128, 64] if not provided.
    lr : float
        Learning rate for the optimiser.  A small value like 1e-4
        works well for on‑line adaptation.
    gamma : float
        Discount factor used in temporal difference targets.
    device : torch.device or None
        Device on which to allocate parameters.  Defaults to CUDA if
        available, else CPU.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Optional[List[int]] = None,
        lr: float = 1e-4,
        gamma: float = 0.99,
        device: Optional[torch.device] = None,
    ) -> None:
        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build actor and critic networks
        self.actor = self._build_network(obs_dim, action_dim, hidden_sizes)
        self.critic = self._build_network(obs_dim, 1, hidden_sizes)

        # Learnable log standard deviation for each action dimension
        self.log_std = nn.Parameter(torch.zeros(action_dim, device=self.device))

        # Optimiser over both networks and log_std
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_std],
            lr=lr,
        )

        # Move networks to the correct device
        self.actor.to(self.device)
        self.critic.to(self.device)

        # Storage for the last state, log probability and value estimate
        self._prev_state: Optional[np.ndarray] = None
        self._prev_logp: Optional[torch.Tensor] = None
        self._prev_value: Optional[torch.Tensor] = None

    def _build_network(self, input_dim: int, output_dim: int, hidden_sizes: List[int]) -> nn.Module:
        """Construct a simple feedforward network with ReLU activations."""
        layers: List[nn.Module] = []
        in_f = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_f, h))
            layers.append(nn.ReLU())
            in_f = h
        layers.append(nn.Linear(in_f, output_dim))
        return nn.Sequential(*layers)

    def load_from_model(self, model: nn.Module) -> None:
        """Initialise actor weights from a pretrained model.

        The provided ``model`` should be a ``torch.nn.Module`` with
        identical architecture (input_dim → hidden_layers → output_dim).
        Only the parameters of the actor network are loaded; the
        critic is left untouched.
        """
        state = model.state_dict()
        actor_state = {k: v for k, v in state.items() if k in self.actor.state_dict()}
        self.actor.load_state_dict(actor_state, strict=False)

    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Sample an action for the given observation.

        Parameters
        ----------
        obs : ndarray
            Observation vector.  Must have length equal to ``obs_dim``.

        Returns
        -------
        action : ndarray
            Sampled action vector (length ``action_dim``) in the same
            format as expected by the Unity environment.  The values are
            typically in a range that maps to joint targets.
        logp : torch.Tensor
            Log probability of the sampled action under the policy.
        value : torch.Tensor
            Critic's value estimate for the given observation.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        # Forward through actor to obtain mean actions
        mean = self.actor(obs_t)  # shape (1, action_dim)
        # Convert log_std to std and expand to match batch
        std = torch.exp(self.log_std).unsqueeze(0)  # shape (1, action_dim)
        # Create independent normal distributions per action dimension
        dist = torch.distributions.Normal(mean, std)
        # Sample from the distribution
        action = dist.sample()  # shape (1, action_dim)
        # Compute log probabilities of the sampled actions
        logp = dist.log_prob(action).sum(dim=-1)  # shape (1,)
        # Critic value estimate for baseline
        value = self.critic(obs_t).squeeze(-1)  # shape (1,)
        # Store for the next update
        self._prev_state = obs.copy()
        self._prev_logp = logp
        self._prev_value = value
        # Return action as numpy array
        return action.detach().cpu().squeeze(0).numpy(), logp.detach(), value.detach()

    def update(self, reward: float, next_obs: Optional[np.ndarray], done: bool = False) -> None:
        """Perform a gradient update using the reward from the last action.

        This method should be called *after* ``select_action`` and after
        receiving the reward signal for the previous step.  It uses
        temporal difference learning to compute an advantage and
        updates both the actor and critic networks.

        Parameters
        ----------
        reward : float
            Reward received for the previous action.  Positive values
            encourage the action; negative values discourage it.
        next_obs : ndarray | None
            Observation for the next state.  May be ``None`` if there
            is no next observation (e.g., at the end of an episode).
        done : bool, default=False
            Whether the episode has ended.  If true, the bootstrap
            value for the next state is treated as zero.
        """
        # If this is the first call or the previous call didn't record state
        if self._prev_state is None or self._prev_logp is None or self._prev_value is None:
            return
        # Compute value of next state for TD target
        if done or next_obs is None:
            next_value = torch.tensor(0.0, device=self.device)
        else:
            next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            next_value = self.critic(next_obs_t).squeeze(-1).detach()
        # Compute TD error and advantage
        td_target = reward + self.gamma * next_value
        advantage = td_target - self._prev_value  # advantage is 1-step TD error
        # Actor loss: negative log probability times advantage (we want to maximise)
        actor_loss = -(self._prev_logp * advantage.detach())
        # Critic loss: mean squared TD error
        critic_loss = advantage.pow(2)
        loss = actor_loss + 0.5 * critic_loss
        # Optimiser step
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to avoid exploding updates (optional but recommended)
        nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_std], max_norm=1.0)
        self.optimizer.step()
        # Reset storage if episode ended
        if done:
            self._prev_state = None
            self._prev_logp = None
            self._prev_value = None
