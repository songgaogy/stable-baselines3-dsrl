import math
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Tuple, List, Type
from dataclasses import dataclass 

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import FlattenExtractor, create_mlp


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
    Distributional V-Network for IQL.
    Estimates V(s) as a categorical distribution.
    Output: Logits of shape (Batch, num_bins)
    """
    def __init__(
        self, 
        observation_space: spaces.Space, 
        hidden_dim: int = 256, 
        depth: int = 3,
        num_bins: int = 51,
        v_min: float = 0.0,
        v_max: float = 1.0
    ):
        super().__init__()
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max
        
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
            
        # Output logits for num_bins
        layers.append(nn.Linear(hidden_dim, num_bins))
        
        self.net = nn.Sequential(*layers)
        
        # Register support vector for calculating mean
        edges = torch.linspace(v_min, v_max, num_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        self.register_buffer("support", centers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns raw LOGITS for V(s).
        Shape: (Batch, num_bins)
        """
        return self.net(obs)
    
    def get_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns probabilities (softmax of logits).
        """
        logits = self.forward(obs)
        return F.softmax(logits, dim=-1)

    def get_v_mean(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns scalar mean of the distribution: E[V] = sum(p_i * z_i).
        Shape: (Batch, 1)
        """
        probs = self.get_probs(obs)
        return torch.sum(probs * self.support, dim=-1, keepdim=True)


class CriticNetwork(ContinuousCritic):
    """
    Distributional Q-Network for IQL (HL-Gauss).
    Outputs LOGITS for categorical distribution bins.
    
    Architecture: [256, 256] -> num_bins
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
        num_bins: int = 51,
        v_min: float = 0.0,
        v_max: float = 1.0,
    ):
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max
        
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
        
        # Re-define q_networks to have `num_bins` output instead of 1
        action_dim = int(np.prod(action_space.shape))
        self.q_networks = []
        for _ in range(self.n_critics):
            q_net_layers = create_mlp(
                input_dim=features_dim + action_dim, 
                output_dim=num_bins,
                net_arch=net_arch, 
                activation_fn=activation_fn
            )
            q_net = nn.Sequential(*q_net_layers)  # 必须封装成 Module
            
            self.q_networks.append(q_net)
        
        self.q_networks = nn.ModuleList(self.q_networks)
        
        # Register support vector for calculating mean
        edges = torch.linspace(v_min, v_max, num_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        self.register_buffer("support", centers)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Returns raw LOGITS for all critics.
        Output Shape: (Batch, num_bins) per critic
        """
        with torch.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
            
        qvalue_input = torch.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def get_probs(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Returns probabilities (softmax of logits).
        """
        logits_tuple = self.forward(obs, actions)
        return tuple(F.softmax(logits, dim=-1) for logits in logits_tuple)

    def get_q_mean(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Returns scalar mean of the distribution: E[Q] = sum(p_i * z_i).
        """
        probs_tuple = self.get_probs(obs, actions)
        return tuple(torch.sum(probs * self.support, dim=-1, keepdim=True) for probs in probs_tuple)