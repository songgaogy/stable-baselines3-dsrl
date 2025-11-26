from typing import Any, Dict, List, Optional, Tuple, Type, Union, TypeVar, Callable
import math
import time
import numpy as np
import torch
from gymnasium import spaces
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import warnings
from collections import deque

from stable_baselines3.common import utils
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TrainFreq, RolloutReturn, TrainFrequencyUnit
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import should_collect_more_steps

SelfDSRLFlow = TypeVar("SelfDSRLFlow", bound="DSRL_Flow")


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
		diffusion_act_dim: Tuple[int, int] = (1, 1), # (chunk_length, action_dim)
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
		self.chunk_length = diffusion_act_dim[0]
		self.act_dim = diffusion_act_dim[1]
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
		raise NotImplementedError("Use DSRLFlow.predict instead")


class DSRL_Flow(OffPolicyAlgorithm):
	"""
	Flow-based Noise Generator for a frozen downstream Diffusion Policy.
	
	Training (BC on v-flow):
		1. Sample (obs, action) from buffer.
		2. Find Target Noise (x_1) such that DP (obs, x_1) approx Action.
		3. Train Flow Model to transport Random Noise (x_0) -> Target Noise (x_1).
	"""
	policy = FlowPolicy

	def __init__(
		self,
		env: Union[GymEnv, str],
		diffusion_policy: Any,
		diffusion_act_dim: Tuple[int, int],
		buffer_size: int = 1_000_000,
		learning_starts: int = 100,
		batch_size: int = 256,
		train_freq: Union[int, Tuple[int, str]] = 1,
		gradient_steps: int = 1,
		action_noise: Optional[ActionNoise] = None,
		replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
		replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
		tensorboard_log: Optional[str] = None,
		policy_kwargs: Optional[Dict[str, Any]] = None,
		verbose: int = 0,
		seed: Optional[int] = None,
		device: Union[torch.device, str] = "auto",
		_init_setup_model: bool = True,
		max_episode_steps: int = 400
	):
		self.cfg = FlowConfig()
		if policy_kwargs is None:
			policy_kwargs = {}
		policy_kwargs["diffusion_act_dim"] = diffusion_act_dim
		# policy_kwargs["observation_space"] = env.observation_space
		# policy_kwargs["action_space"] = env.action_space
		# policy_kwargs["lr_schedule"] = self.cfg.lr_schedule

		super().__init__(
			policy=FlowPolicy,
			env=env,
			learning_rate=self.cfg.learning_rate,
			buffer_size=buffer_size,
			learning_starts=learning_starts,
			batch_size=batch_size,
			tau=0.0,                # dummy params, not used
			gamma=0.0,              # dummy params, not used
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

		self.diffusion_policy = diffusion_policy
		self.diffusion_act_chunk = diffusion_act_dim[0]
		self.diffusion_act_dim = diffusion_act_dim[1]
		self.max_episode_steps = max_episode_steps
		self.env_buffers = None
		self.submission_staging = None
		
		if _init_setup_model:
			self._setup_model()

	def _setup_model(self) -> None:
		super()._setup_model()
		
		self.policy = self.policy_class(
			self.observation_space,
			self.action_space,      # TODO(gaoyuan) check this!
			self.cfg.lr_schedule,
			**self.policy_kwargs,
		)
		self.policy = self.policy.to(self.device)
		self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr_schedule(1))

	def train(self, gradient_steps: int, batch_size: int = 64) -> None:
		"""
		V-Flow Training Loop.
		"""
		self.policy.set_training_mode(True)
		self._update_learning_rate(self.optimizer)
		losses = []

		for _ in range(gradient_steps):
			# sample, all good
			replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
			obs = replay_data.observations
			noise_action = replay_data.noise_actions    # NOTE(gaoyuan) modify `type_aliases.py`, in shape (chunk*dim)
			obs = replay_data.observations
			target_noise = noise_action.view(batch_size, self.diffusion_act_chunk, self.diffusion_act_dim)

			x_1 = target_noise              # expected noise for flow output = DP input
			x_0 = torch.randn_like(x_1)     # input noise for flow

			t = torch.rand(batch_size, device=self.device)
			t_b = t.view(batch_size, 1, 1)
			x_t = (1 - t_b) * x_0 + t_b * x_1

			v_target = x_1 - x_0
			v_pred = self.policy(obs, x_t, t)

			loss = F.mse_loss(v_pred, v_target)
			losses.append(loss.item())

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

		self._n_updates += gradient_steps
		self.logger.record("train/loss", np.mean(losses))

	def predict(
		self,
		observation: Union[np.ndarray, Dict[str, np.ndarray]],
		episode_start: Optional[np.ndarray] = None,
		deterministic: bool = False,
	) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
		self.policy.set_training_mode(False)
		observation, vectorized_env = self.policy.obs_to_tensor(observation)
		batch_size = observation.shape[0]

		with torch.no_grad():
			x = torch.randn(
				batch_size,
				self.diffusion_act_chunk,
				self.diffusion_act_dim,
				device=self.device,
			)
			
			num_steps = self.cfg.flow_steps
			dt = 1.0 / num_steps
			
			for i in range(num_steps):
				t_val = i * dt
				t_tensor = torch.full((batch_size,), t_val, device=self.device)
				
				v_pred = self.policy(observation, x, t_tensor)
				x = x + v_pred * dt

			refined_noise = x       # estimated x_1 (input noise for DP)

			# Downstream Policy Inference
			action = self.diffusion_policy(
				observation, 
				refined_noise, 
				return_numpy=False
			)
			
			action = action.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)
			action = action.cpu().numpy()

		if isinstance(self.action_space, spaces.Box):
			# clip based on environment bounds, not arbitrary -1,1
			action = np.clip(action, self.action_space.low, self.action_space.high)

		return action, refined_noise
	
	def learn(
		self: SelfDSRLFlow,
		total_timesteps: int,
		callback: MaybeCallback = None,
		log_interval: int = 4,
		tb_log_name: str = "FlowAgent",
		reset_num_timesteps: bool = True,
		progress_bar: bool = False,
	) -> SelfDSRLFlow:
		"""
		NOTE(gaoyuan) This method is originally provided by stable_baseline3 source code; But since 
		we make quite distinct modification, we have to rewrite `.learn()` method. The overal logic 
		and code structure is copied from stable_baseline3
		"""
		replay_buffer = self.replay_buffer
		truncate_last_traj = (
			self.optimize_memory_usage
			and reset_num_timesteps
			and replay_buffer is not None
			and (replay_buffer.full or replay_buffer.pos > 0)
		)

		if truncate_last_traj:
			warnings.warn(
				"The last trajectory in the replay buffer will be truncated, "
				"see https://github.com/DLR-RM/stable-baselines3/issues/46."
				"You should use `reset_num_timesteps=False` or `optimize_memory_usage=False`"
				"to avoid that issue."
			)
			assert replay_buffer is not None
			pos = (replay_buffer.pos - 1) % replay_buffer.buffer_size
			replay_buffer.dones[pos] = True

		assert self.env is not None, "You must set the environment before calling _setup_learn()"
		
		self.start_time = time.time_ns()

		if self.ep_info_buffer is None or reset_num_timesteps:
			# Initialize buffers if they don't exist, or reinitialize if resetting counters
			self.ep_info_buffer = deque(maxlen=self._stats_window_size)
			self.ep_success_buffer = deque(maxlen=self._stats_window_size)

		if self.action_noise is not None:
			self.action_noise.reset()

		if reset_num_timesteps:
			self.num_timesteps = 0
			self._episode_num = 0
		else:
			total_timesteps += self.num_timesteps
		self._total_timesteps = total_timesteps
		self._num_timesteps_at_start = self.num_timesteps

		# avoid resetting the environment when calling `.learn()` consecutive times
		if reset_num_timesteps or self._last_obs is None:
			assert self.env is not None
			self._last_obs = self.env.reset()  # type: ignore[assignment]
			self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
			# Retrieve unnormalized observation for saving into the buffer
			if self._vec_normalize_env is not None:
				self._last_original_obs = self._vec_normalize_env.get_original_obs()

		# configure logger's outputs if no logger was passed
		if not self._custom_logger:
			self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

		callback = self._init_callback(callback, progress_bar)

		callback.on_training_start(locals(), globals())
		assert self.env is not None, "You must set the environment before calling learn()"
		assert isinstance(self.train_freq, TrainFreq)  # check done in _setup_learn()

		while self.num_timesteps < total_timesteps:
			rollout = self.collect_rollouts(
				self.env,
				train_freq=self.train_freq,
				action_noise=self.action_noise,
				callback=callback,
				learning_starts=self.learning_starts,
				replay_buffer=self.replay_buffer,
				log_interval=log_interval,
			)

			if not rollout.continue_training:
				break

			if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
				# If no `gradient_steps` is specified,
				# do as many gradients steps as steps performed during the rollout
				gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
				# Special case when the user passes `gradient_steps=0`
				if gradient_steps > 0:
					self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
					current_progress = 1.0 - (self.num_timesteps / total_timesteps)
					self._current_progress_remaining = current_progress
					self._update_learning_rate(self.optimizer)  

		callback.on_training_end()

		return self
	
	def collect_rollouts(
		self,
		env: VecEnv,
		callback: BaseCallback,
		train_freq: TrainFreq,
		replay_buffer: ReplayBuffer,
		action_noise: Optional[ActionNoise] = None,
		learning_starts: int = 0,
		log_interval: Optional[int] = None,
	) -> RolloutReturn:
		"""
		Collect experiences and store them into a ``ReplayBuffer``.

		:param env: The training environment
		:param callback: Callback that will be called at each step
			(and at the beginning and end of the rollout)
		:param train_freq: How much experience to collect
			by doing rollouts of current policy.
			Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
			or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
			with ``<n>`` being an integer greater than 0.
		:param action_noise: Action noise that will be used for exploration
			Required for deterministic policy (e.g. TD3). This can also be used
			in addition to the stochastic policy for SAC.
		:param learning_starts: Number of steps before learning for the warm-up phase.
		:param replay_buffer:
		:param log_interval: Log data every ``log_interval`` episodes
		"""
		self.policy.set_training_mode(False)
		num_collected_steps, num_collected_episodes = 0, 0
		assert isinstance(env, VecEnv), "You must pass a VecEnv"
		assert train_freq.frequency > 0, "Should at least collect one step or episode."

		if env.num_envs > 1:
			assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

		callback.on_rollout_start()
		n_envs = self.replay_buffer.n_envs
		buffer_n_envs = replay_buffer.n_envs
		
		# Initialize persistent staging if not present
		if self.env_buffers is None or len(self.env_buffers) != n_envs:
			self.env_buffers = [[] for _ in range(n_envs)]
		if self.submission_staging is None:
			self.submission_staging = []

		# success if reward > -reward_offset (e.g., > -1.0 for offset 1.0)
		reward_offset = 1.0 
		success_threshold = -reward_offset
		_warned_about_success = False

		# collection Loop
		while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
			obs_tensor = torch.as_tensor(self._last_obs, device=self.device, dtype=torch.float32)
			batch_size = self.replay_buffer.n_envs
			
			with torch.no_grad():
				# Base Noise (Prior)
				x = torch.randn(
					batch_size, 
					self.diffusion_act_chunk, 
					self.diffusion_act_dim, 
					device=self.device
				)
				
				# Flow ODE Solver
				num_steps = self.cfg.flow_steps
				dt = 1.0 / num_steps
				for i in range(num_steps):
					t_val = i * dt
					t_tensor = torch.full((batch_size,), t_val, device=self.device)
					v_pred = self.policy(obs_tensor, x, t_tensor)
					x = x + v_pred * dt
				
				refined_noise = x # Estimated x_1
				
				# diffusion policy
				action = self.diffusion_policy(obs_tensor, refined_noise, return_numpy=False)
				action = action.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)
				action_np = action.cpu().numpy()
				
				if isinstance(self.action_space, spaces.Box):
					action_np = np.clip(action_np, self.action_space.low, self.action_space.high)

			# Step Environment
			new_obs, rewards, dones, infos = env.step(action_np)
			self.num_timesteps += n_envs
			
			for i in range(n_envs):
				real_next_obs = new_obs[i]
				if dones[i]:
					if infos[i] and "terminal_observation" in infos[i]:
						real_next_obs = infos[i]["terminal_observation"]
				
				noise_np = refined_noise[i].cpu().numpy().reshape(-1)

				transition = (
					self._last_obs[i].copy(),
					real_next_obs.copy(),
					action_np[i].copy(),
					rewards[i],
					dones[i],
					infos[i],
					noise_np
				)
				
				# Add to persistent per-env buffer
				self.env_buffers[i].append(transition)

				if dones[i]:
					is_success = False
					# Matches 'reward > -rew_offset' logic
					if rewards[i] > success_threshold:
						is_success = True

					should_save = is_success
					
					if should_save:
						# Move to persistent staging
						self.submission_staging.extend(self.env_buffers[i])
						num_collected_episodes += 1
						self._episode_num += 1

						# Process Staging
						while len(self.submission_staging) >= buffer_n_envs:
							batch = self.submission_staging[:buffer_n_envs]
							# Update the persistent staging buffer
							self.submission_staging = self.submission_staging[buffer_n_envs:]
							
							b_obs, b_nobs, b_act, b_rew, b_done, b_info, b_noise = zip(*batch)
							
							replay_buffer.add(
								obs=np.stack(b_obs),
								next_obs=np.stack(b_nobs),
								action=np.stack(b_act),
								reward=np.stack(b_rew),
								done=np.stack(b_done),
								infos=list(b_info),
								noise_action=np.stack(b_noise)
							)
							num_collected_steps += buffer_n_envs
					
					# clear env buffer for this env only
					self.env_buffers[i] = []

			self._last_obs = new_obs
			
			# standard SB3 callback
			callback.update_locals(locals())
			if not callback.on_step():
				return RolloutReturn(num_collected_steps, num_collected_episodes, False)

		callback.on_rollout_end()
		return RolloutReturn(num_collected_steps, num_collected_episodes, True)

	def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
		return ["policy", "optimizer"], []
