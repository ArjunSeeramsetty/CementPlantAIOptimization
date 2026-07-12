"""
Custom PyTorch Actor-Critic PPO Agent for continuous Kiln-Cooler process control.
Ensures lightweight dependencies and native execution inside the platform.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.distributions import Normal
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

class Actor(nn.Module):
    """Policy network outputting action distributions (mean and log_std)."""
    def __init__(self, state_dim: int, action_dim: int):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.mean_layer = nn.Linear(64, action_dim)
        # Log standard deviation parameter for continuous policy exploration
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean = self.mean_layer(x)
        # Standard deviation must be positive
        std = torch.exp(self.log_std)
        return mean, std


class Critic(nn.Module):
    """Value network predicting the state value V(s)."""
    def __init__(self, state_dim: int):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class PPOMemory:
    """Buffer to store transitions for a training update cycle."""
    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.is_terminals: List[bool] = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.is_terminals.clear()


class PPOAgent:
    """Proximal Policy Optimization agent optimized for continuous control zones."""
    def __init__(self, state_dim: int = 7, action_dim: int = 3, lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.memory = PPOMemory()

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Choose action continuous parameters under Gaussian policy.
        
        Returns:
            action, log_prob, state_value
        """
        state_t = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            mean, std = self.actor(state_t)
            value = self.critic(state_t)
            
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
        return action.cpu().numpy(), log_prob.item(), value.item()

    def update(self, clip_eps: float = 0.2, ppo_epochs: int = 10, mini_batch_size: int = 32) -> Tuple[float, float]:
        """
        Train the networks using stored memory experiences.
        
        Returns:
            Average Actor Loss, Average Critic Loss
        """
        # Convert lists to PyTorch tensors
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.memory.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory.log_probs).to(self.device)
        old_values = torch.FloatTensor(self.memory.values).to(self.device)
        rewards = self.memory.rewards
        is_terminals = self.memory.is_terminals
        
        # Calculate Returns and Advantages using GAE (Generalized Advantage Estimation)
        returns = []
        discounted_sum = 0
        
        # Loop backwards to calculate GAE
        advantages = torch.zeros(len(rewards) + 1, dtype=torch.float32).to(self.device)
        gae = 0
        for i in reversed(range(len(rewards))):
            next_value = 0 if i == len(rewards) - 1 else old_values[i + 1]
            non_terminal = 1.0 - float(is_terminals[i])
            delta = rewards[i] + self.gamma * next_value * non_terminal - old_values[i]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[i] = gae
            
        returns_tensor = advantages[:-1] + old_values
        advantages = (advantages[:-1] - advantages[:-1].mean()) / (advantages[:-1].std() + 1e-8)
        
        actor_losses = []
        critic_losses = []
        
        dataset_size = len(states)
        
        # PPO surrogate loss gradient descent loops
        for _ in range(ppo_epochs):
            permutation = torch.randperm(dataset_size)
            for start_idx in range(0, dataset_size, mini_batch_size):
                batch_indices = permutation[start_idx : start_idx + mini_batch_size]
                
                b_states = states[batch_indices]
                b_actions = actions[batch_indices]
                b_old_log_probs = old_log_probs[batch_indices]
                b_returns = returns_tensor[batch_indices]
                b_advantages = advantages[batch_indices]
                
                # Forward passes
                mean, std = self.actor(b_states)
                values = self.critic(b_states).squeeze()
                
                dist = Normal(mean, std)
                log_probs = dist.log_prob(b_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
                
                # Ratio of probabilities
                ratios = torch.exp(log_probs - b_old_log_probs)
                
                # Clipped surrogate objective loss
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
                
                # Value loss (Mean Squared Error)
                critic_loss = nn.MSELoss()(values, b_returns)
                
                # Optimizers updates
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                
        # Reset memory buffer
        self.memory.clear()
        
        return np.mean(actor_losses), np.mean(critic_losses)

    def save(self, filepath: str):
        """Save the actor and critic state dicts."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, filepath)
        logger.info("Saved PPO agent weights to: %s", filepath)

    def load(self, filepath: str) -> bool:
        """Load state dicts from file."""
        if not os.path.exists(filepath):
            return False
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            logger.info("Loaded PPO agent weights from: %s", filepath)
            return True
        except Exception as e:
            logger.warning("Failed to load agent checkpoint: %s", e)
            return False
