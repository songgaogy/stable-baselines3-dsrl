import math
from gymnasium import spaces
import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Tuple, List, Type
from dataclasses import dataclass 

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import FlattenExtractor


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        # progress_remaining: 1 -> 0
        return initial_value * progress_remaining
    return func


@dataclass
class FlowConfig:
    flow_steps: int = 100
    learning_rate: float = 1e-4
    lr_schedule: Callable = linear_schedule(learning_rate)


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal timestep embedding for the Flow Network.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FlowPolicy(BasePolicy):
    """
    A time-conditioned MLP that models the velocity field v(x_t, t | obs).
    Outputs NOISE for the downstream diffusion policy.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        act_dim: Tuple[int, int] = (1, 1), # (chunk_length, action_dim)
        net_arch: list = [256, 256, 256], 
        activation_fn: type[nn.Module] = nn.GELU,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class=None,
            optimizer_class=torch.optim.Adam,
        )
        self.lr_schedule = lr_schedule
        self.chunk_length = act_dim[0]
        self.act_dim = act_dim[1]
        self.total_noise_dim = self.chunk_length * self.act_dim

        obs_dim = observation_space.shape[0]
        assert isinstance(obs_dim, int)
        
        # Timestep embedding dimension
        time_dim = 64
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Input: Obs + Noisy Noise Input (x_t) + Time Embedding
        input_dim = obs_dim + self.total_noise_dim + time_dim   # 23 + 28 + 64 = 115
        
        layers = []
        prev_dim = input_dim
        for layer_dim in net_arch['pi']:
            layers.append(nn.Linear(prev_dim, layer_dim))
            layers.append(activation_fn())
            prev_dim = layer_dim
        
        # Output: Velocity (same shape as noise)
        layers.append(nn.Linear(prev_dim, self.total_noise_dim))
        
        self.flow_net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity v_theta(x_t, t, obs)
        
        :param obs: Observation [B, obs_dim]
        :param x_t: Current noise estimate at time t [B, chunk, dim]
        :param t: Timestep [B]
        """
        batch_size = obs.shape[0]
        
        t_embed = self.time_mlp(t)  # [B, time_dim]
        x_t_flat = x_t.view(batch_size, -1) 
        inputs = torch.cat([obs, x_t_flat, t_embed], dim=1).float()
        
        velocity = self.flow_net(inputs)
        return velocity.view(batch_size, self.chunk_length, self.act_dim)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        raise NotImplementedError("Use FLOW.predict instead")


class ValueNetwork(nn.Module):
    """
    V-Network for IQL (Expectile Regression).
    Estimates V(s).
    """
    def __init__(
        self, 
        observation_space: spaces.Space, 
        hidden_dim: int = 256, 
        depth: int = 3
    ):
        super().__init__()
        # Handle observation dimension (assuming Box/Flat for now)
        if isinstance(observation_space, spaces.Box):
            input_dim = int(np.prod(observation_space.shape))
        else:
            raise NotImplementedError("ValueNetwork currently only supports Box observation spaces.")

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CriticNetwork(ContinuousCritic):
    """
    Q-Network for IQL.
    Inherits from SB3 ContinuousCritic to leverage Double-Q logic 
    and feature extraction support.
    
    Default Architecture: [256, 256]
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        net_arch: List[int] = [256, 256, 256],
        features_extractor: nn.Module = None,
        features_dim: int = None,
        activation_fn: Type[nn.Module] = nn.GELU,
        normalize_images: bool = True,
        share_features_extractor: bool = True,
    ):
        if features_extractor is None:
            features_extractor = FlattenExtractor(observation_space)

        if features_dim is None:
            features_dim = features_extractor.features_dim

        super().__init__(
            observation_space,
            action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
            share_features_extractor=share_features_extractor,
        )
