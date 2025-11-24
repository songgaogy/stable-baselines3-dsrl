from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy

from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv

SelfDSRL = TypeVar("SelfDSRL", bound="DSRL")


class DSRL(OffPolicyAlgorithm):
	"""
	DSRL-NA (noise aliased variant of DSRL)
	Based on the SAC implementation in Stable Baselines3.
	Paper: https://arxiv.org/pdf/2506.15799

	:param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
	:param env: The environment to learn from (if registered in Gym, can be str)
	:param learning_rate: learning rate for adam optimizer,
		the same learning rate will be used for all networks (Q-Values, Actor and Value function)
		it can be a function of the current progress remaining (from 1 to 0)
	:param buffer_size: size of the replay buffer
	:param learning_starts: how many steps of the model to collect transitions for before learning starts
	:param batch_size: Minibatch size for each gradient update
	:param tau: the soft update coefficient ("Polyak update", between 0 and 1)
	:param gamma: the discount factor
	:param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
		like ``(5, "step")`` or ``(2, "episode")``.
	:param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
		Set to ``-1`` means to do as many gradient steps as steps done in the environment
		during the rollout.
	:param action_noise: the action noise type (None by default), this can help
		for hard exploration problem. Cf common.noise for the different action noise type.
	:param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
		If ``None``, it will be automatically selected.
	:param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
	:param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
		at a cost of more complexity.
		See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
	:param ent_coef: Entropy regularization coefficient. (Equivalent to
		inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
		Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
	:param target_update_interval: update the target network every ``target_network_update_freq``
		gradient steps.
	:param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
	:param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
		instead of action noise exploration (default: False)
	:param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
		Default: -1 (only sample at the beginning of the rollout)
	:param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
		during the warm up phase (before learning starts)
	:param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
		the reported success rate, mean episode length, and mean reward over
	:param tensorboard_log: the log location for tensorboard (if None, no logging)
	:param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`sac_policies`
	:param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
		debug messages
	:param seed: Seed for the pseudo random generators
	:param device: Device (cpu, cuda, ...) on which the code should be run.
		Setting it to auto, the code will be run on the GPU if possible.
	:param _init_setup_model: Whether or not to build the network at the creation of the instance
	:param actor_gradient_steps: Number of gradient steps to take on actor per training update
	:param diffusion_policy: The diffusion policy to use for action generation
	:param diffusion_act_dim: The action dimension for the diffusion policy (tuple of (action chunk length, action_dim))
	:param noise_critic_grad_steps: Number of gradient steps to take on distilled noise critic per training update
	:param critic_backup_combine_type: How to combine the critics for the backup (min or mean)
	"""
	policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
		"MlpPolicy": MlpPolicy,	# MlpPolicy == SACPolicy
		"CnnPolicy": CnnPolicy,
		"MultiInputPolicy": MultiInputPolicy,
	}
	policy: SACPolicy
	actor: Actor
	critic: ContinuousCritic
	critic_target: ContinuousCritic
	critic_noise: ContinuousCritic

	def __init__(
		self,
		policy: Union[str, type[SACPolicy]],
		env: Union[GymEnv, str],
		learning_rate: Union[float, Schedule] = 3e-4,
		buffer_size: int = 1_000_000,  # 1e6
		learning_starts: int = 100,
		batch_size: int = 256,
		tau: float = 0.005,
		gamma: float = 0.99,
		train_freq: Union[int, tuple[int, str]] = 1,
		gradient_steps: int = 1,
		action_noise: Optional[ActionNoise] = None,
		replay_buffer_class: Optional[type[ReplayBuffer]] = None,
		replay_buffer_kwargs: Optional[dict[str, Any]] = None,
		optimize_memory_usage: bool = False,
		ent_coef: Union[str, float] = "auto",
		target_update_interval: int = 1,
		target_entropy: Union[str, float] = "auto",
		use_sde: bool = False,
		sde_sample_freq: int = -1,
		use_sde_at_warmup: bool = False,
		stats_window_size: int = 100,
		tensorboard_log: Optional[str] = None,
		policy_kwargs: Optional[dict[str, Any]] = None,
		verbose: int = 0,
		seed: Optional[int] = None,
		device: Union[torch.device, str] = "auto",
		_init_setup_model: bool = True,
		actor_gradient_steps: int = -1,
		diffusion_policy=None,
		diffusion_act_dim=None,
		noise_critic_grad_steps: int = 1,
		critic_backup_combine_type='min',
	):
		super().__init__(
			policy,
			env,
			learning_rate,
			buffer_size,
			learning_starts,
			batch_size,
			tau,
			gamma,
			train_freq,
			gradient_steps,
			action_noise,
			replay_buffer_class=replay_buffer_class,
			replay_buffer_kwargs=replay_buffer_kwargs,
			policy_kwargs=policy_kwargs,
			stats_window_size=stats_window_size,
			tensorboard_log=tensorboard_log,
			verbose=verbose,
			device=device,
			seed=seed,
			use_sde=use_sde,
			sde_sample_freq=sde_sample_freq,
			use_sde_at_warmup=use_sde_at_warmup,
			optimize_memory_usage=optimize_memory_usage,
			supported_action_spaces=(spaces.Box,),
			support_multi_env=True,
		)

		self.target_entropy = target_entropy
		self.log_ent_coef = None  # type: Optional[torch.Tensor]
		# Entropy coefficient / Entropy temperature
		# Inverse of the reward scale
		self.ent_coef = ent_coef
		self.target_update_interval = target_update_interval
		self.ent_coef_optimizer: Optional[torch.optim.Adam] = None
		self.actor_gradient_steps = actor_gradient_steps
		
		self.diffusion_policy = diffusion_policy
		self.diffusion_act_chunk = diffusion_act_dim[0]
		self.diffusion_act_dim = diffusion_act_dim[1]
		self.noise_critic_grad_steps = noise_critic_grad_steps
		self.critic_backup_combine_type = critic_backup_combine_type

		if _init_setup_model:
			self._setup_model()

	def _setup_model(self) -> None:
		super()._setup_model()
		self._create_aliases()	# (gaoyuan) all policy / actor refer to noise space

		# Running mean and running var
		self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
		self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

		# Target entropy is used when learning the entropy coefficient
		if self.target_entropy == "auto":
			# automatically set target entropy if needed
			self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))
		else:
			# Force conversion
			# this will also throw an error for unexpected string
			self.target_entropy = float(self.target_entropy)

		# The entropy coefficient or entropy can be learned automatically
		# see Automating Entropy Adjustment for Maximum Entropy RL section
		# of https://arxiv.org/abs/1812.05905
		if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
			# Default initial value of ent_coef when learned
			init_value = 1.0
			if "_" in self.ent_coef:
				init_value = float(self.ent_coef.split("_")[1])
				assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

			# Note: we optimize the log of the entropy coeff which is slightly different from the paper
			# as discussed in https://github.com/rail-berkeley/softlearning/issues/37
			self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * init_value).requires_grad_(True)
			self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
		else:
			# Force conversion to float
			# this will throw an error if a malformed string (different from 'auto')
			# is passed
			self.ent_coef_tensor = torch.tensor(float(self.ent_coef), device=self.device)
		
		policy_noise = self.policy_class(	# SACPolicy
			self.observation_space,
			self.action_space,
			self.lr_schedule,
			**self.policy_kwargs,
		)
		self.critic_noise = policy_noise.critic		# (gaoyuan) only get critic part
		self.critic_noise: ContinuousCritic = self.critic_noise.to(self.device)		# Q_w(s, a) in noise space

	def _create_aliases(self) -> None:
		self.actor = self.policy.actor
		self.critic = self.policy.critic
		self.critic_target = self.policy.critic_target

	def train(self, gradient_steps: int, batch_size: int = 64) -> None:
		# Switch to train mode (this affects batch norm / dropout)
		self.policy.set_training_mode(True)
		self.critic_noise.set_training_mode(True)
		# Update optimizers learning rate
		optimizers = [self.actor.optimizer, self.critic.optimizer, self.critic_noise.optimizer]
		if self.ent_coef_optimizer is not None:
			optimizers += [self.ent_coef_optimizer]

		# Update learning rate according to lr schedule
		self._update_learning_rate(optimizers)

		ent_coef_losses, ent_coefs = [], []
		actor_losses, critic_losses, noise_critic_losses = [], [], []

		if self.actor_gradient_steps < 0:
			actor_gradient_idx = np.linspace(0, gradient_steps-1, gradient_steps, dtype=int)
		else:
			actor_gradient_idx = np.linspace(int(gradient_steps / self.actor_gradient_steps) - 1, gradient_steps-1, self.actor_gradient_steps, dtype=int)

		for gradient_step in range(gradient_steps):
			# Sample replay buffer
			# (gaoyuan) `replay_data.actions` stores actual action in env but not sampled noise
			replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

			# We need to sample because `log_std` may have changed between two gradient steps
			if self.use_sde:
				self.actor.reset_noise()

			# Action by the current actor for the sampled state
			actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
			log_prob = log_prob.reshape(-1, 1)

			ent_coef_loss = None
			if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
				# Important: detach the variable from the graph
				# so we don't change it with other losses
				# see https://github.com/rail-berkeley/softlearning/issues/60
				ent_coef = torch.exp(self.log_ent_coef.detach())
				ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
				ent_coef_losses.append(ent_coef_loss.item())
			else:
				ent_coef = self.ent_coef_tensor

			ent_coefs.append(ent_coef.item())

			# Optimize entropy coefficient, also called entropy temperature or alpha in the paper
			if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
				self.ent_coef_optimizer.zero_grad()
				ent_coef_loss.backward()
				self.ent_coef_optimizer.step()

			with torch.no_grad():
				# Select action according to policy
				next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
				next_actions = torch.tensor(self.policy.unscale_action(next_actions.cpu().numpy())).to(self.device)
				next_actions = self.diffusion_policy(	# (gaoyuan) DP: (real_obs, noise) -> real_action
					replay_data.next_observations, 
					next_actions.reshape(-1, self.diffusion_act_chunk, self.diffusion_act_dim), 
					return_numpy=False
				)
				next_actions = next_actions.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)

				# Compute the next Q values: min over all critics targets
				# (gaoyuan) here Q is in real physical world
				next_q_values = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
				if self.critic_backup_combine_type == 'min':
					next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
				elif self.critic_backup_combine_type == 'mean':
					next_q_values = torch.mean(next_q_values, dim=1, keepdim=True)
				# add entropy term
				next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
				# td error + entropy term
				target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

			# Get current Q-values estimates for each critic network
			# using action from the replay buffer
			current_q_values = self.critic(replay_data.observations, replay_data.actions)

			# Compute critic loss: in real-world Q_A(s, a)
			critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
			assert isinstance(critic_loss, torch.Tensor)  # for type checker
			critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

			# Optimize the critic
			self.critic.optimizer.zero_grad()
			critic_loss.backward()
			self.critic.optimizer.step()

			if gradient_step in actor_gradient_idx:
				# Compute actor loss
				# Alternative: actor_loss = torch.mean(log_prob - qf1_pi)
				# Min over all critic networks
				q_values_pi = torch.cat(self.critic_noise(replay_data.observations, actions_pi), dim=1)
				if self.critic_backup_combine_type == 'min':
					min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
				elif self.critic_backup_combine_type == 'mean':
					min_qf_pi = torch.mean(q_values_pi, dim=1, keepdim=True)
				actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
				actor_losses.append(actor_loss.item())

				# Optimize the actor
				self.actor.optimizer.zero_grad()
				actor_loss.backward()
				self.actor.optimizer.step()

			# Update target networks
			if gradient_step % self.target_update_interval == 0:
				polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
				# Copy running stats, see GH issue #996
				polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

		# (gaoyuan) distillation, for `noise_critic` in noise space
		for gradient_step in range(self.noise_critic_grad_steps):
			# Sample replay buffer
			critic_distill_loss = 0
			replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
			critic_distill_loss = critic_distill_loss + self.update_noise_critic(replay_data)
			noise_critic_losses.append(critic_distill_loss.item())
			self.critic_noise.optimizer.zero_grad()
			critic_distill_loss.backward()
			self.critic_noise.optimizer.step()

		self.critic_noise.set_training_mode(False)
		self._n_updates += gradient_steps

		self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
		self.logger.record("train/ent_coef", np.mean(ent_coefs))
		self.logger.record("train/actor_loss", np.mean(actor_losses))
		self.logger.record("train/critic_loss", np.mean(critic_losses))
		self.logger.record("train/noise_critic_loss", np.mean(noise_critic_losses))
		if len(ent_coef_losses) > 0:
			self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

	def update_noise_critic(self, replay_data):
		with torch.no_grad():
			noise_actions = torch.randn(	# [B, chunk_size, action_dim]
				replay_data.actions.shape[0], 
				self.diffusion_act_chunk, 
				self.diffusion_act_dim
			).to(self.device)
			diffused_actions = self.diffusion_policy(replay_data.observations, noise_actions, return_numpy=False)
			diffused_actions = diffused_actions.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)
			current_q_values = self.critic(replay_data.observations, diffused_actions)

		# NOTE(gaoyuan) 
		noise_actions = noise_actions.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim).detach().cpu().numpy()
		noise_actions = torch.tensor(self.policy.scale_action(noise_actions)).to(self.device)
		current_q_noise_vals = self.critic_noise(replay_data.observations, noise_actions)
		critic_distill_loss = 0
		
		for i in range(len(current_q_values)):
			current_q = current_q_values[i]
			current_q_noise = current_q_noise_vals[i]
			critic_distill_loss = critic_distill_loss + 0.5 * F.mse_loss(current_q.detach(), current_q_noise)
		return critic_distill_loss

	def learn(
		self: SelfDSRL,
		total_timesteps: int,
		callback: MaybeCallback = None,
		log_interval: int = 4,
		tb_log_name: str = "SAC",
		reset_num_timesteps: bool = True,
		progress_bar: bool = False,
	) -> SelfDSRL:
		return super().learn(
			total_timesteps=total_timesteps,
			callback=callback,
			log_interval=log_interval,
			tb_log_name=tb_log_name,
			reset_num_timesteps=reset_num_timesteps,
			progress_bar=progress_bar,
		)

	def _excluded_save_params(self) -> list[str]:
		return super()._excluded_save_params() + ["actor", "critic", "critic_target"]  # noqa: RUF005

	def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
		state_dicts = ["policy", "actor.optimizer", "critic.optimizer", "critic_noise"]
		if self.ent_coef_optimizer is not None:
			saved_pytorch_variables = ["log_ent_coef"]
			state_dicts.append("ent_coef_optimizer")
		else:
			saved_pytorch_variables = ["ent_coef_tensor"]
		# import pdb; pdb.set_trace()
		return state_dicts, saved_pytorch_variables
	
	def _sample_action(
		self,
		learning_starts: int,
		action_noise: Optional[ActionNoise] = None,
		n_envs: int = 1,
	) -> tuple[np.ndarray, np.ndarray]:
		"""
		Sample an action according to the exploration policy.
		This is either done by sampling the probability distribution of the policy,
		or sampling a random action (from a uniform distribution over the action space)
		or by adding noise to the deterministic output.

		:param action_noise: Action noise that will be used for exploration
			Required for deterministic policy (e.g. TD3). This can also be used
			in addition to the stochastic policy for SAC.
		:param learning_starts: Number of steps before learning for the warm-up phase.
		:param n_envs:
		:return: action to take in the environment
			and scaled action that will be stored in the replay buffer.
			The two differs when the action space is not normalized (bounds are not [-1, 1]).
		"""
		# Select action randomly or according to policy
		# (gaoyuan) directly sample action in real-world
		if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
			# Warmup phase
			unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
		else:
			# Note: when using continuous actions,
			# we assume that the policy uses tanh to scale the action
			# We use non-deterministic action in the case of SAC, for TD3, it does not matter
			assert self._last_obs is not None, "self._last_obs was not set"
			unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

		# Rescale the action from [low, high] to [-1, 1]
		if isinstance(self.action_space, spaces.Box):
			scaled_action = self.policy.scale_action(unscaled_action)

			# Add noise to the action (improve exploration)
			if action_noise is not None:
				scaled_action = np.clip(scaled_action + action_noise(), -1, 1)	# NOTE: here we clip the action!

			# We store the scaled action in the buffer
			buffer_action = scaled_action
			action = self.policy.unscale_action(scaled_action)
		else:
			# Discrete case, no need to normalize or clip
			buffer_action = unscaled_action
			action = buffer_action

		action: torch.Tensor = torch.as_tensor(action, device=self.device, dtype=torch.float32)
		obs = torch.as_tensor(self._last_obs, device=self.device, dtype=torch.float32)

		action = self.diffusion_policy(obs, action.reshape(-1, self.diffusion_act_chunk, self.diffusion_act_dim), return_numpy=False)
		action = action.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)
		action = action.cpu().numpy()
		buffer_action = action
		return action, buffer_action

	def predict_diffused(
		self,
		observation: Union[np.ndarray, dict[str, np.ndarray]],
		state: Optional[tuple[np.ndarray, ...]] = None,
		episode_start: Optional[np.ndarray] = None,
		deterministic: bool = False,
	) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
		"""(gaoyuan)
		Given an observation, predict `action_noise`, scale and unscale, then pass through DP to get real action
		"""
		unscaled_action, predict_second_return = self.policy.predict(observation, state, episode_start, deterministic)
		if isinstance(self.action_space, spaces.Box):
			scaled_action = self.policy.scale_action(unscaled_action)
			# We store the scaled action in the buffer
			buffer_action = scaled_action
			action = self.policy.unscale_action(scaled_action)
		else:
			# Discrete case, no need to normalize or clip
			buffer_action = unscaled_action
			action = buffer_action

		action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
		obs = torch.as_tensor(observation, device=self.device, dtype=torch.float32)
		action = self.diffusion_policy(obs, action.reshape(-1, self.diffusion_act_chunk, self.diffusion_act_dim), return_numpy=False)
		action = action.reshape(-1, self.diffusion_act_chunk * self.diffusion_act_dim)
		action = action.cpu().numpy()
		return action, predict_second_return
