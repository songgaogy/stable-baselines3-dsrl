import math
import time
import copy
import numpy as np
import torch
import pathlib
from gymnasium import spaces
import torch.nn as nn
from torch.nn import functional as F
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Type, Union, TypeVar

from stable_baselines3.common import utils
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.utils import polyak_update, expectile_loss
from stable_baselines3.common.save_util import load_from_pkl

from .networks import FlowConfig, FlowPolicy, CriticNetwork, ValueNetwork

SelfFLOW = TypeVar("SelfFLOW", bound="FLOW")



class FLOW(OffPolicyAlgorithm):
    """
    FLOW: IQL Critics + Flow-Matching Policy (Offline RL).
    
    Phases:
    1. BC Phase (`learn_bc_iql`): Updates policy using Behavior Cloning.
    2. Dipole Phase (`learn`): Updates Q/V (IQL) and Policy (NFT/Advantage).
    """
    policy: FlowPolicy
    policy_old: FlowPolicy
    critic: CriticNetwork
    critic_target: CriticNetwork
    value: ValueNetwork

    def __init__(
        self,
        env: Union[GymEnv, str],
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        act_dim: Tuple[int, int] = None,
        expectile: float = 0.9,
        temperature: float = 3.0,
        policy_eta: float = 1.0,
        critic_eta: float = 0.1,
        guidance_w: List[float] = [0.0, 1.0],
        best_of_n: List[int] = [1, 2, 4],
        beta: float = 1.0,
        bc_buffer: str = "success",
        discount: float = 0.99, 
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        max_episode_steps: int = 400,
    ):
        self.cfg = FlowConfig()

        if policy_kwargs is None:
            policy_kwargs = {}
        policy_kwargs["act_dim"] = act_dim

        super().__init__(
            policy=FlowPolicy,
            env=env,
            learning_rate=self.cfg.learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=0.005,      # redundant
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

        self.expectile = expectile
        self.temperature = temperature
        self.policy_eta = policy_eta
        self.critic_eta = critic_eta

        # for dipole
        self.beta = beta
        self.guidance_w = guidance_w
        self.bc_buffer = bc_buffer
        self.discount = discount
        self.best_of_n = best_of_n

        self.max_episode_steps = max_episode_steps
        self.env_buffers = None
        self.submission_staging = None

        self._n_updates = 0
        
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(FLOW, self)._setup_model()
        
        self.policy_pos = self.policy_class(
            self.observation_space,
            self.action_space,
            self.cfg.lr_schedule,
            **self.policy_kwargs,
        ).to(self.device)
        
        self.policy_neg = self.policy_class(
            self.observation_space,
            self.action_space,
            self.cfg.lr_schedule,
            **self.policy_kwargs,
        ).to(self.device)

        # for stable_baseline3 compability
        self.policy = self.policy_pos
        self.actor = self.policy_pos

        features_extractor = self.policy.features_extractor if hasattr(self.policy, "features_extractor") else None
        
        self.critic = CriticNetwork(
            self.observation_space,
            self.action_space,
            features_extractor=features_extractor,
            net_arch=[256, 256]
        ).to(self.device)
        
        self.critic_target = CriticNetwork(
            self.observation_space,
            self.action_space,
            features_extractor=features_extractor,
            net_arch=[256, 256]
        ).to(self.device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.set_training_mode(False)

        self.value = ValueNetwork(
            self.observation_space, 
            hidden_dim=256
        ).to(self.device)

        self.policy_pos_optimizer = torch.optim.Adam(self.policy_pos.parameters(), lr=1e-4)
        self.policy_neg_optimizer = torch.optim.Adam(self.policy_neg.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=1e-4)
        
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

        if reset_num_timesteps:     # when BC, do reset
            self.num_timesteps = 0
            self._episode_num = 0
        else:                       # when NFT, do NOT reset
            total_timesteps += self.num_timesteps   # Make sure training timesteps are ahead of the internal counter
        
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
        use_01_reward: bool = False,
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

    def learn_bc_iql(
        self: SelfFLOW,
        iterations: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "dipole",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfFLOW:
        """
        Behavior Cloning to warmup NFT.  
        Trains the flow policy to match the dataset distribution (Offline).
        """
        self.eval_freq = callback[1].eval_freq
        total_timesteps, callback = self._setup_learn(
            iterations, 
            callback, 
            reset_num_timesteps, 
            tb_log_name=tb_log_name, 
            progress_bar=progress_bar
        )
        callback.on_training_start(locals(), globals())
        print(f"[FLOW] Starting BC Phase for {iterations} steps...")
        print(f"[INFO] eval freq: {self.eval_freq}")

        while self.num_timesteps < total_timesteps:
            # BC for policys
            self.policy_pos.set_training_mode(True)
            self.policy_neg.set_training_mode(True)
            if self.bc_buffer == "success":
                self._train_bc_step(buffer=self.succ_buffer, batch_size=self.batch_size)
            elif self.bc_buffer == "all":
                self._train_bc_step(buffer=self.replay_buffer, batch_size=self.batch_size)
            else:
                raise NotImplementedError("invalid buffer type!")
            self.policy_pos.set_training_mode(False)
            self.policy_neg.set_training_mode(False)

            # IQL for Q and V
            self.critic.set_training_mode(True)
            self.value.train(True)
            self._train_qv_iql_step(buffer=self.replay_buffer, batch_size=self.batch_size)
            self.critic.set_training_mode(False)
            self.value.train(False)
            
            self.num_timesteps += 1
            self._n_updates += 1

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

    def learn(
        self: SelfFLOW,
        iterations: int, # NOTE: This refers to gradient steps in offline RL
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "dipole",
        reset_num_timesteps: bool = False, # False to continue from BC
        progress_bar: bool = False,
    ) -> SelfFLOW:
        """
        NFT (Negative Flow Tuning) + IQL (Value Learning).
        Updates Critics (Q/V) and refines Policy using Advantage.
        """
        # 1. Setup (Continue from previous steps if reset=False)
        eval_freq = callback[1].eval_freq
        total_timesteps, callback = self._setup_learn(
            iterations, 
            callback, 
            reset_num_timesteps,    # do not reset, but continue
            tb_log_name=tb_log_name,
            progress_bar=progress_bar
        )
        callback.on_training_start(locals(), globals())
        print(f"[FLOW] Starting DIPOLE Phase for {iterations} steps...")
        print(f"[INFO] eval freq: {eval_freq}")

        while self.num_timesteps < total_timesteps:
            # perform IQL updates first to get accurate Advantage
            self.critic.set_training_mode(True)
            self.value.train(True)
            self.policy.set_training_mode(False)
            self._train_qv_iql_step(buffer=self.replay_buffer, batch_size=self.batch_size)
            
            # dual policy updates
            self.critic.set_training_mode(False)
            self.value.train(False)
            self.policy_pos.set_training_mode(True)
            self.policy_neg.set_training_mode(True)
            self._train_dual_net_step(buffer=self.replay_buffer, batch_size=self.batch_size)

            # Update counters
            self.num_timesteps += 1
            self._n_updates += 1

            # Logging
            if self.num_timesteps % log_interval == 0:
                self.logger.record("eval/global_step", self.num_timesteps)
                self.logger.record("train/global_step", self.num_timesteps)
                self._dump_logs()

            # Callback (Evaluation / Checkpointing)
            callback.update_locals(locals())
            if not callback.on_step():
                break

        callback.on_training_end()
        return self

    def _train_bc_step(self, buffer: ReplayBuffer, batch_size: int) -> None:
        """
        Single gradient step for Behavior Cloning, for policy only
        """
        replay_data = buffer.sample(batch_size, env=self._vec_normalize_env)
        obs = replay_data.observations
        actions = replay_data.actions
        
        # we expected actions in shape: (B, chunk_len, act_dim)
        assert len(actions.shape) == 2, f"actions in replay buffer has no expected shape, with shape of: {actions.shape}"
        target_actions = actions.view(batch_size, self.policy.chunk_length, self.policy.act_dim).to(self.device)

        x_1 = target_actions
        x_0 = torch.randn_like(x_1, device=self.device)
        t = torch.rand(batch_size, device=self.device)
        t_expand = t.view(batch_size, 1, 1) if x_1.dim() == 3 else t.view(batch_size, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        v_target = x_1 - x_0

        # Update Policy Pos
        v_pred_pos = self.policy_pos(obs, x_t, t)
        loss_pos = F.mse_loss(v_pred_pos, v_target)

        self.policy_pos_optimizer.zero_grad()
        loss_pos.backward()
        self.policy_pos_optimizer.step()

        # Update Policy Neg
        v_pred_neg = self.policy_neg(obs, x_t, t)
        loss_neg = F.mse_loss(v_pred_neg, v_target)
        
        self.policy_neg_optimizer.zero_grad()
        loss_neg.backward()
        self.policy_neg_optimizer.step()

        self.logger.record("train/bc_loss_pos", loss_pos.item())
        self.logger.record("train/bc_loss_neg", loss_neg.item())

    def _train_qv_iql_step(self, buffer: ReplayBuffer, batch_size: int) -> None:
        """
        Single gradient step for IQL (Q and V functions)
        """
        replay_data = buffer.sample(batch_size, env=self._vec_normalize_env)
        obs = replay_data.observations
        actions = replay_data.actions
        next_obs = replay_data.next_observations
        rewards = replay_data.rewards
        dones = replay_data.dones

        # Value Loss (Expectile Regression)
        with torch.no_grad():
            # NOTE(gaoyuan) actions are flattened
            target_q1, target_q2 = self.critic_target(obs, actions)
            target_q = torch.min(target_q1, target_q2)

        v_pred = self.value(obs)
        v_loss = expectile_loss(target_q - v_pred, self.expectile)

        self.value_optimizer.zero_grad()
        v_loss.backward()
        self.value_optimizer.step()

        # Critic Loss
        with torch.no_grad():
            next_v = self.value(next_obs)
            target_q_values = rewards + (1 - dones) * self.gamma * next_v

        current_q1, current_q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(current_q1, target_q_values) + F.mse_loss(current_q2, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Soft update target critic
        if self._n_updates % 1 == 0:
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.critic_eta)
        
        self.logger.record("train/value_loss", v_loss.item())
        self.logger.record("train/critic_loss", critic_loss.item())
        self.logger.record("train/v_pred_mean", v_pred.mean().item())

    def _train_dual_net_step(self, buffer: ReplayBuffer, batch_size: int, nft_beta: float = 1.0) -> None:
        """
        Single gradient step for NFT
        """
        replay_data = buffer.sample(batch_size, env=self._vec_normalize_env)
        obs = replay_data.observations
        actions = replay_data.actions
        next_obs = replay_data.next_observations

        # advantage
        with torch.no_grad():
            # Calculate V_current and V_next
            v_current = self.value(obs)
            v_next = self.value(next_obs)
            
            # (gaoyuan) follow the instruction of ZhiHao :)
            guidance_term = (2.0 - self.discount) * v_next - v_current
            
            # weights for NFT
            weights_pos = torch.sigmoid(guidance_term)
            weights_neg = 1.0 - weights_pos
        
        # Flow Matching Setup
        x_1 = actions.view(batch_size, self.policy_pos.chunk_length, self.policy_pos.act_dim)
        
        # Expand weights to match action dims [B, 1, 1]
        weights_pos_exp = weights_pos.view(batch_size, 1, 1)
        weights_neg_exp = weights_neg.view(batch_size, 1, 1)

        x_0 = torch.randn_like(x_1)
        t = torch.rand(batch_size, device=self.device)
        t_expand = t.view(batch_size, 1, 1) if x_1.dim() == 3 else t.view(batch_size, 1)

        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1
        v_target = x_1 - x_0

        # Update Positive Policy (theta_1)
        v_pred_pos = self.policy_pos(obs, x_t, t)
        loss_pos = torch.mean(weights_pos_exp * (v_pred_pos - v_target)**2)
        
        self.policy_pos_optimizer.zero_grad()
        loss_pos.backward()
        self.policy_pos_optimizer.step()

        # Update Negative Policy (theta_2)
        v_pred_neg = self.policy_neg(obs, x_t, t)
        loss_neg = torch.mean(weights_neg_exp * (v_pred_neg - v_target)**2)

        self.policy_neg_optimizer.zero_grad()
        loss_neg.backward()
        self.policy_neg_optimizer.step()

        self.logger.record("train/dual_loss_pos", loss_pos.item())
        self.logger.record("train/dual_loss_neg", loss_neg.item())
        self.logger.record("train/G_mean", guidance_term.mean().item())
        self.logger.record("train/weight_pos_mean", weights_pos.mean().item())
        self.logger.record("train/weight_neg_mean", weights_neg.mean().item())

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        w: float = 1.0
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Sampling according to papers
        """
        assert self.device.type == 'cuda', f"Device Assertion Failed: Expected 'cuda', but got '{self.device}'."
        
        # Ensure eval mode
        self.policy_pos.set_training_mode(False)
        self.policy_neg.set_training_mode(False)
        
        observation, vectorized_env = self.policy_pos.obs_to_tensor(observation)
        batch_size = observation.shape[0]

        with torch.no_grad():
            x = torch.randn(
                batch_size,
                self.policy_pos.chunk_length, 
                self.policy_pos.act_dim,
                device=self.device,
            )
            
            num_steps = self.cfg.flow_steps
            dt = 1.0 / num_steps
            
            for i in range(num_steps):
                t_val = i * dt
                t_tensor = torch.full((batch_size,), t_val, device=self.device)
                
                v_pos = self.policy_pos(observation, x, t_tensor)
                v_neg = self.policy_neg(observation, x, t_tensor)
                
                # according to papers
                v_pred = (1 + w) * v_pos - w * v_neg
                
                x = x + v_pred * dt

            action = x.reshape(-1, self.policy_pos.chunk_length * self.policy_pos.act_dim)
            action = action.cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            action = np.clip(action, self.action_space.low, self.action_space.high)

        return action

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        return [
            "policy_pos", 
            "policy_neg", 
            "critic", 
            "value", 
            "policy_pos_optimizer", 
            "policy_neg_optimizer", 
            "critic_optimizer", 
            "value_optimizer"
        ], []
