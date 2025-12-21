import os
import time
import math
import pathlib
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from gymnasium import spaces
from torch.nn import functional as F

from typing import Any, Dict, List, Optional, Tuple, Type, Union, TypeVar

from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common import utils
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import polyak_update, expectile_loss
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.policies import BasePolicy
from .networks import CriticNetwork, ValueNetwork

SelfDist_Q = TypeVar("SelfDist_Q", bound="Dist_Q")



class HLGaussLoss(nn.Module):
    """
    Histogram Loss - Gaussian (HL-Gauss) for Distributional RL.

    (gaoyuan) refer to Stop Regressing paper: https://arxiv.org/pdf/2403.03950
    """
    def __init__(self, min_value: float, max_value: float, num_bins: int, sigma_ratio: float = 0.75):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.sigma_ratio = sigma_ratio
        
        self.register_buffer(
            "support_edges", 
            torch.linspace(min_value, max_value, num_bins + 1)
        )
        
        self.bin_width = (max_value - min_value) / num_bins
        self.sigma = sigma_ratio * self.bin_width

    def _transform_to_probs(self, target_scalar: torch.Tensor) -> torch.Tensor:
        if target_scalar.ndim == 1:
            target_scalar = target_scalar.unsqueeze(-1)

        target = target_scalar.clamp(self.min_value, self.max_value)
        normalized_x = (self.support_edges - target) / (self.sigma * math.sqrt(2))
        cdf_evals = 0.5 * (1 + torch.erf(normalized_x))
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = bin_probs / (z.unsqueeze(-1) + 1e-6)
        return bin_probs

    def forward(self, logits: torch.Tensor, target_scalar: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes Cross Entropy Loss.
        If weights is provided, computes weighted mean loss (used for Expectile regression in V).
        """
        with torch.no_grad():
            target_probs = self._transform_to_probs(target_scalar)

        ce_loss = F.cross_entropy(logits, target_probs, reduction='none')
        
        if weights is not None:     # Ensure weights shape matches loss shape [Batch]
            weights = weights.view(-1)
            loss = (ce_loss * weights).mean()
        else:
            loss = ce_loss.mean()
            
        return loss



class Dist_Q(OffPolicyAlgorithm):
    """
    IQL (Implicit Q-Learning) Implementation.
    Includes Critic (Q), Value (V), and Actor (Policy) updates.

    NOTE(Gaoyuan) here we add policy in order to evaluate Q
    """
    critic: CriticNetwork
    critic_target: CriticNetwork
    value: ValueNetwork
    policy: SACPolicy  # Explicit type hint

    def __init__(
        self,
        env: Union[GymEnv, str],
        policy: Union[str, Type[BasePolicy]] = SACPolicy,
        learning_rate: float = 1e-4,
        buffer_size: int = 10_000_000,
        learning_starts: int = 1,
        batch_size: int = 256,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        act_dim: Tuple[int, int] = None,
        expectile: float = 0.9,
        temperature: float = 3.0,
        critic_eta: float = 0.01,
        buffer: str = "success",
        discount: float = 0.99,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        max_episode_steps: int = 400,
        num_bins: int = 101,
        v_min: float = -10.0,
        v_max: float = 60.0,
        sigma_ratio: float = 0.75,  # be the same as original paper
    ):
        if policy_kwargs is None:
            policy_kwargs = {}
        if act_dim is not None:
             policy_kwargs["act_dim"] = act_dim

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=critic_eta,         # redundant
            gamma=discount,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_sample_freq=-1,
            use_sde=False,
            use_sde_at_warmup=False,
            optimize_memory_usage=False,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )
        # for IQL
        self.expectile = expectile
        self.temperature = temperature
        self.critic_eta = critic_eta
        self.buffer = buffer
        self.max_episode_steps = max_episode_steps

        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max
        self.sigma_ratio = sigma_ratio
        
        self.env_buffers = None
        self._n_updates = 0
        
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        # using the inner actor from `OffPolicyAlgorithm` class
        super(Dist_Q, self)._setup_model()
        features_extractor = self.policy.features_extractor if hasattr(self.policy, "features_extractor") else None
        
        # NOTE: distributional
        self.critic = CriticNetwork(
            self.observation_space,
            self.action_space,
            features_extractor=features_extractor,
            net_arch=[256, 256],
            num_bins=self.num_bins,
            v_min=self.v_min,
            v_max=self.v_max
        ).to(self.device)
        
        self.critic_target = CriticNetwork(
            self.observation_space,
            self.action_space,
            features_extractor=features_extractor,
            net_arch=[256, 256],
            num_bins=self.num_bins,
            v_min=self.v_min,
            v_max=self.v_max
        ).to(self.device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.set_training_mode(False)

        # NOTE: distributional
        self.value = ValueNetwork(
            self.observation_space, 
            hidden_dim=256,
            num_bins=self.num_bins,
            v_min=self.v_min,
            v_max=self.v_max
        ).to(self.device)

        self.hl_gauss_loss = HLGaussLoss(
            min_value=self.v_min, 
            max_value=self.v_max, 
            num_bins=self.num_bins, 
            sigma_ratio=self.sigma_ratio
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.learning_rate)

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        """
        Custom _setup_learn for Offline RL.
        Overrides the standard SB3 setup to avoid:
        1. Checking for self.env (we might be pure offline)
        2. Resetting the env (unnecessary for offline)
        3. Truncating the replay buffer (destructive for static datasets)
        """
        self.start_time = time.time_ns()

        if self.ep_info_buffer is None or reset_num_timesteps:
            self.ep_info_buffer = deque(maxlen=self._stats_window_size)
            self.ep_success_buffer = deque(maxlen=self._stats_window_size)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps
        
        # Configure logger
        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Initialize callback
        callback = self._init_callback(callback, progress_bar)

        return total_timesteps, callback

    def load_replay_buffers(
        self,
        directory: Union[str, pathlib.Path],
        target_buffer: ReplayBuffer,
        prefix: str = "success_data",
        size: int = 2000,
        use_01_reward: bool = True,
        reward_offset: int = 1,
        truncate_last_traj: bool = True,
        verbose: int = 1,
    ) -> None:
        """(gaoyuan)
        Load multiple replay buffer .pkl files from a directory and
        append their transitions into a target buffer (e.g. succ_buffer or all_buffer).

        This uses the same loading logic as `load_replay_buffer` / `load_from_pkl`
        from SB3, but does not overwrite `self.replay_buffer`.

        :param directory: Directory containing pickled replay buffers.
        :param target_buffer: The buffer to which transitions will be appended.
        :param prefix: Filename prefix to match, e.g. "success_data".
        :param truncate_last_traj: Same meaning as in `load_replay_buffer` for HerReplayBuffer.
        :param verbose: Verbosity level.
        """
        directory = pathlib.Path(directory)
        pattern = f"{prefix}*.pkl"
        pkl_paths = sorted(directory.glob(pattern))

        if verbose > 0:
            print(f"[load_multiple_replay_buffers] Found {len(pkl_paths)} files with pattern '{pattern}' in {directory}")

        total_added = 0

        for path in pkl_paths:
            if verbose > 0:
                print(f"[load_multiple_replay_buffers] Loading {path.name}")

            # Use the same logic as original load_from_pkl
            loaded_buffer = load_from_pkl(path, verbose=verbose)
            assert isinstance(
                loaded_buffer, ReplayBuffer
            ), "The loaded object must inherit from ReplayBuffer class"

            if not hasattr(loaded_buffer, "handle_timeout_termination"):  # pragma: no cover
                loaded_buffer.handle_timeout_termination = False
                loaded_buffer.timeouts = np.zeros_like(loaded_buffer.dones)

            if isinstance(loaded_buffer, HerReplayBuffer):
                assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"
                loaded_buffer.set_env(self.env)
                if truncate_last_traj:
                    loaded_buffer.truncate_last_trajectory()

            # Match device to current setting
            loaded_buffer.device = self.device

            # for 0/1 reward
            success_threshold = -reward_offset

            # Append transitions from loaded_buffer into target_buffer
            n_transitions = loaded_buffer.size()
            assert size <= 5000
            load_transitions = int(size / 5000 * n_transitions)
            for idx in range(load_transitions):
                obs = loaded_buffer.observations[idx]
                next_obs = loaded_buffer.next_observations[idx]
                actions = loaded_buffer.actions[idx]
                rewards = loaded_buffer.rewards[idx]
                if use_01_reward:   # convert to 0/1 reward: 1 if reward > threshold, else 0
                    rewards = (rewards > success_threshold).astype(rewards.dtype)
                dones = loaded_buffer.dones[idx]
                infos = [{} for _ in range(self.n_envs)]

                target_buffer.add(obs, next_obs, actions, rewards, dones, infos)
                total_added += 1

        if verbose > 0:
            print(
                f"[load_multiple_replay_buffers] Done. "
                f"Loaded {len(pkl_paths)} files, appended {total_added} transitions to target buffer."
            )

    def learn(
        self: SelfDist_Q,
        iterations: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "IQL",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        save_path: Optional[str] = None,
        total_save_num: int = 0,
    ) -> SelfDist_Q:
        """
        main entrance for IQL
        """
        eval_freq = callback[1].eval_freq
        total_timesteps, callback = self._setup_learn(
            iterations, 
            callback, 
            reset_num_timesteps, 
            tb_log_name=tb_log_name, 
            progress_bar=progress_bar
        )
        callback.on_training_start(locals(), globals())
        print(f"[IQL] Starting training for {iterations} steps...")
        print(f"[INFO] eval freq: {eval_freq}")

        save_interval = float('inf')
        if total_save_num > 0:
            save_interval = max(1, iterations // total_save_num)
            print(f"[IQL] Model will be saved every {save_interval} steps to {save_path}")

        while self.num_timesteps < total_timesteps:
            # IQL for Q and V
            self.critic.set_training_mode(True)
            self.value.train(True)
            self.policy.set_training_mode(True)

            if self.buffer == "success":
                self._train_iql_step(buffer=self.succ_buffer, batch_size=self.batch_size)
            elif self.buffer == "all":
                self._train_iql_step(buffer=self.replay_buffer, batch_size=self.batch_size)
            else:
                raise NotImplementedError
            
            self.critic.set_training_mode(False)
            self.value.train(False)
            self.policy.set_training_mode(False)
            
            self.num_timesteps += 1
            self._n_updates += 1

            if save_path is not None and total_save_num > 0:
                if self.num_timesteps % save_interval == 0:
                    # Create directory if it doesn't exist
                    if not os.path.exists(save_path):
                        os.makedirs(save_path, exist_ok=True)
                    
                    save_file = os.path.join(save_path, f"model_{self.num_timesteps}_steps")
                    self.save(save_file)
                    if self.verbose > 0:
                        print(f"Saved model to {save_file}")

            if self.num_timesteps % log_interval == 0:
                self.logger.record("eval/global_step", self.num_timesteps)
                self.logger.record("train/global_step", self.num_timesteps)
                self._dump_logs()
            
            # Callback (Evaluation / Checkpointing)
            callback.update_locals(locals())
            # NOTE(gaoyuan) evaluation starts here
            if not callback.on_step():  # n_calls += 1
                break

        callback.on_training_end()
        return self

    def _train_iql_step(self, buffer: ReplayBuffer, batch_size: int) -> None:
        replay_data = buffer.sample(batch_size, env=self._vec_normalize_env)
        obs = replay_data.observations
        actions = replay_data.actions
        next_obs = replay_data.next_observations
        rewards = replay_data.rewards
        dones = replay_data.dones

        # Update Value (V) Distribution
        with torch.no_grad():
            target_q1_mean, target_q2_mean = self.critic_target.get_q_mean(obs, actions)    # scaler outputs
            target_q_mean = torch.min(target_q1_mean, target_q2_mean)
            v_mean = self.value.get_v_mean(obs)     # scaler outputs
            
            diff = target_q_mean - v_mean
            weights = torch.where(diff > 0, self.expectile, 1 - self.expectile)

        v_logits = self.value(obs)
        v_loss = self.hl_gauss_loss(v_logits, target_q_mean, weights)

        self.value_optimizer.zero_grad()
        v_loss.backward()
        self.value_optimizer.step()

        # Update Critic (Q) Distribution
        with torch.no_grad():
            # use the MEAN of the updated V distribution.
            next_v_mean = self.value.get_v_mean(next_obs)
            target_q_scalar = rewards + (1 - dones) * self.gamma * next_v_mean

        current_q1_logits, current_q2_logits = self.critic(obs, actions)
        
        loss_q1 = self.hl_gauss_loss(current_q1_logits, target_q_scalar)
        loss_q2 = self.hl_gauss_loss(current_q2_logits, target_q_scalar)
        critic_loss = loss_q1 + loss_q2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor (Policy)
        with torch.no_grad():
             current_q1_mean, current_q2_mean = self.critic.get_q_mean(obs, actions)    # use mean, same as original IQL
             current_q_mean = torch.min(current_q1_mean, current_q2_mean)
             v_mean_detached = self.value.get_v_mean(obs).detach()
        
        advantage = current_q_mean - v_mean_detached
        exp_adv = torch.exp(self.temperature * advantage).clamp(max=100.0)
        
        mean_actions, log_std, kwargs = self.policy.actor.get_action_dist_params(obs)
        dist = self.policy.actor.action_dist.proba_distribution(mean_actions, log_std)
        log_prob = dist.log_prob(actions)
        if len(log_prob.shape) == 1:
            log_prob = log_prob.reshape(-1, 1)
        
        actor_loss = -(exp_adv * log_prob).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self._n_updates % 1 == 0:
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.critic_eta)
        
        self.logger.record("train/value_loss", v_loss.item())
        self.logger.record("train/critic_loss", critic_loss.item())
        self.logger.record("train/actor_loss", actor_loss.item())
        self.logger.record("train/q_target_mean", target_q_scalar.mean().item())
        self.logger.record("train/advantage_mean", advantage.mean().item())

    def get_q_value(self, observation: np.ndarray, action: np.ndarray) -> np.ndarray:
        self.critic.eval()
        with torch.no_grad():
            obs_tensor = torch.as_tensor(observation, device=self.device)
            act_tensor = torch.as_tensor(action, device=self.device)
            q1_mean, q2_mean = self.critic.get_q_mean(obs_tensor, act_tensor)
            q_min = torch.min(q1_mean, q2_mean)
        return q_min.cpu().numpy()
    
    # def predict(
    #     self,
    #     observation: Union[np.ndarray, Dict[str, np.ndarray]],
    #     state: Optional[Tuple[np.ndarray, ...]] = None,
    #     episode_start: Optional[np.ndarray] = None,
    #     deterministic: bool = False,
    # ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
    #     """
    #     Use the learned IQL policy to predict actions.
    #     """
    #     return self.policy.predict(
    #         observation, 
    #         state=state, 
    #         episode_start=episode_start, 
    #         deterministic=deterministic
    #     )
    
    def get_q_value(self, observation: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Returns the Q-value for given state-action pairs.
        """
        self.critic.eval()
        with torch.no_grad():
            obs_tensor = torch.as_tensor(observation, device=self.device)
            act_tensor = torch.as_tensor(action, device=self.device)
            
            q1, q2 = self.critic(obs_tensor, act_tensor)
            q_min = torch.min(q1, q2)
        return q_min.cpu().numpy()

    def save(self, path: Union[str, pathlib.Path, int], **kwargs) -> None:
        data = {
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "value_state_dict": self.value.state_dict(),
            "policy_state_dict": self.policy.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "policy_optimizer_state_dict": self.actor_optimizer.state_dict(),
        }
        save_to_pkl(path, data, verbose=self.verbose)

    @classmethod
    def load(cls, path, env=None, device="auto", **kwargs):
        loaded_object = load_from_pkl(path, verbose=False)
        if isinstance(loaded_object, dict):
            data = loaded_object
        else:
             data = loaded_object[0] # assuming tuple
        model = cls(env=env, device=device, _init_setup_model=True, **kwargs)
        if "critic_state_dict" in data:
            model.critic.load_state_dict(data["critic_state_dict"])
            model.critic_target.load_state_dict(data["critic_target_state_dict"])
        if "value_state_dict" in data:
            model.value.load_state_dict(data["value_state_dict"])
        if "policy_state_dict" in data:
            model.policy.load_state_dict(data["policy_state_dict"])
        return model
    