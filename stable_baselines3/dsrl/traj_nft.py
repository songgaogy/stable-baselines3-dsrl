import re
import math
import time
import copy
import numpy as np
import torch
import pathlib
from tqdm import tqdm
from omegaconf import ListConfig
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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.utils import polyak_update, expectile_loss
from stable_baselines3.common.save_util import load_from_pkl

from .networks import FlowConfig, FlowPolicy, CriticNetwork, ValueNetwork
from sources.collect_init_data import _add_traj_to_buffer

SelfTrajNFT = TypeVar("SelfTrajNFT", bound="TrajNFT")



class TrajNFT(OffPolicyAlgorithm):
    """
    FLOW: IQL Critics + Flow-Matching Policy (Offline RL).
    
    Phases:
    1. BC Phase (`learn_bc_iql`): Updates policy using Behavior Cloning.
    2. online NFT Phase (`learn`): Updates Q/V (IQL) and Policy (NFT/Advantage), and occasionally collect
        online data using current actor
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
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        debug: bool = False,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        max_episode_steps: int = 400,
        steps_per_iter: int = 10000,
        collecting_num: int = 1,
        nft_delta: float = 1.0,
        with_scale: bool = True,
        sample_mode: str = "half_succ_fail",
        reward_mode: str = "01",
        update_freq: int = 1000,
        succ_warmup_traj: int = 10
    ):
        self.cfg = FlowConfig()
        self.act_chunk = act_dim[0]
        self.action_dim = act_dim[1]

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
            gamma=0.99,
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
        self.debug = debug

        self.steps_per_iter = steps_per_iter
        self.collecting_num = collecting_num
        self.nft_delta = nft_delta
        self.with_scale = with_scale
        self.sample_mode = sample_mode
        self.reward_mode = reward_mode
        self.update_freq = update_freq
        self.succ_warmup_traj = succ_warmup_traj

        self.expectile = expectile
        self.temperature = temperature
        self.policy_eta = policy_eta
        self.critic_eta = critic_eta

        self.max_episode_steps = max_episode_steps
        self.env_buffers = None
        self.submission_staging = None

        self._n_updates = 0
        
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(TrajNFT, self)._setup_model()
        
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.cfg.lr_schedule,
            **self.policy_kwargs,
        ).to(self.device)
        assert isinstance(self.policy, FlowPolicy)
        self.actor = self.policy

        self.policy_old = copy.deepcopy(self.policy)
        
        # freeze policy_old and set to eval mode
        self.policy_old.set_training_mode(False)
        self.policy_old.to(self.device)
        for param in self.policy_old.parameters():
            param.requires_grad = False

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

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
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
        num_trajs: Optional[int] = None,
        truncate_last_traj: bool = True,
        verbose: int = 1,
    ) -> None:
        """(gaoyuan)
        Load replay buffer files. 
        It first scans all files to verify total availability.
        Then it loads files sequentially until `num_trajs` is reached.
        For the last file needed, it loads a fraction based on the remaining count.
        """
        directory = pathlib.Path(directory)
        pattern = f"{prefix}*.pkl"
        pkl_paths = sorted(directory.glob(pattern))

        if len(pkl_paths) == 0:
            print(f"[load_multiple_replay_buffers] No files found with pattern '{pattern}' in {directory}")
            return

        # pre-scan
        file_meta = [] # Stores (path, traj_count)
        total_available_trajs = 0
        
        for path in pkl_paths:
            match = re.search(r"trajs_(\d+)", path.name)    # match for filename "trajs_1000"
            if match:
                count = int(match.group(1))
            else:
                if verbose > 0:
                    print(f"[Warning] Filename {path.name} does not match 'trajs_X'. Skipping trajectory count check for this file.")
                count = 0 # Fallback or handle as error depending on strictness
            
            file_meta.append((path, count))
            total_available_trajs += count

        if verbose > 0:
            print(f"[Pre-scan] Found {len(pkl_paths)} files. Total available trajectories on disk: {total_available_trajs}")

        # Check if requested amount is feasible
        if num_trajs is not None and num_trajs > total_available_trajs:
            print(f"[Warning] Requested {num_trajs} trajs, but only {total_available_trajs} are available. Will load ALL.")
            num_trajs = total_available_trajs # Cap it at max available

        # loading loop
        total_collected_trajs = 0
        total_added_transitions = 0

        for path, file_traj_count in file_meta:
            if num_trajs is not None and total_collected_trajs >= num_trajs:
                break

            load_full_file = True
            trajs_to_take_from_this_file = file_traj_count

            if num_trajs is not None:
                needed = num_trajs - total_collected_trajs
                if needed < file_traj_count:
                    load_full_file = False
                    trajs_to_take_from_this_file = needed
                else:
                    load_full_file = True
                    trajs_to_take_from_this_file = file_traj_count

            if verbose > 0:
                mode_str = "FULL" if load_full_file else f"PARTIAL ({trajs_to_take_from_this_file}/{file_traj_count})"
                print(f"[Loading] {path.name} | Mode: {mode_str}")

            # Load the heavy data
            loaded_buffer = load_from_pkl(path, verbose=0) # keep inner verbose low
            
            assert isinstance(loaded_buffer, ReplayBuffer), "Object must be ReplayBuffer"
            if not hasattr(loaded_buffer, "handle_timeout_termination"):
                loaded_buffer.handle_timeout_termination = False
                loaded_buffer.timeouts = np.zeros_like(loaded_buffer.dones)
            if isinstance(loaded_buffer, HerReplayBuffer):
                loaded_buffer.set_env(self.env)
                if truncate_last_traj:
                    loaded_buffer.truncate_last_trajectory()
            loaded_buffer.device = self.device

            n_transitions_in_file = loaded_buffer.size()
            if load_full_file:
                steps_to_load = n_transitions_in_file
            else:
                if file_traj_count > 0:
                    ratio = trajs_to_take_from_this_file / file_traj_count
                    steps_to_load = int(n_transitions_in_file * ratio)
                else:
                    steps_to_load = n_transitions_in_file

            for idx in range(steps_to_load):
                obs = loaded_buffer.observations[idx]
                next_obs = loaded_buffer.next_observations[idx]
                actions = loaded_buffer.actions[idx]
                rewards = loaded_buffer.rewards[idx]
                dones = loaded_buffer.dones[idx]
                infos = [{} for _ in range(self.n_envs)]

                target_buffer.add(obs, next_obs, actions, rewards, dones, infos)
                total_added_transitions += 1

            total_collected_trajs += trajs_to_take_from_this_file
            
            if verbose > 0:
                print(f"    -> Added {steps_to_load} steps (approx {trajs_to_take_from_this_file} trajs). Progress: {total_collected_trajs}/{num_trajs if num_trajs else 'ALL'}")

        if verbose > 0:
            print(
                f"[load_multiple_replay_buffers] Finished. "
                f"Total Trajs: {total_collected_trajs}, Total Steps: {total_added_transitions}."
            )

    def _online_collection(self, succ_buffer: ReplayBuffer, fail_buffer: ReplayBuffer):
        if self.collecting_num == 0:
            print(f"\n[INFO] offline. No online collection.")
            return 
        else:
            print(f"\n[INFO] start collection {self.collecting_num} trajecotries")
            collect_online_data(
                env=self.env, 
                succ_buffer=succ_buffer, 
                fail_buffer=fail_buffer,
                num_trajectories=self.collecting_num, 
                policy=self.policy,
                chunk_size= self.act_chunk,
                action_dim=self.action_dim,
                device=self.device,
                flow_steps=self.cfg.flow_steps
            )

    def learn_bc(
        self: SelfTrajNFT,
        iterations: int,
        callback: MaybeCallback = None,
        log_interval: int = 200,
        tb_log_name: str = "traj_nft",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTrajNFT:
        """
        Behavior Cloning to warmup NFT.  
        Trains the flow policy to match the dataset distribution (Offline).
        """
        eval_callback = callback[1]
        original_eval_freq = eval_callback.eval_freq
        if not self.debug:
            if iterations > 100000:
                eval_callback.eval_freq = int(original_eval_freq / 10)
                bc_eval_freq = eval_callback.eval_freq
            else:
                eval_callback.eval_freq = int(iterations / 2)
                bc_eval_freq = eval_callback.eval_freq
        else:
            bc_eval_freq = original_eval_freq

        total_timesteps, callback = self._setup_learn(
            iterations, 
            callback, 
            reset_num_timesteps, 
            tb_log_name=tb_log_name, 
            progress_bar=progress_bar
        )
        callback.on_training_start(locals(), globals())
        print(f"[FLOW] Starting BC Phase for {iterations} steps...")
        print(f"[INFO] BC eval freq: {bc_eval_freq}")

        while self.num_timesteps < total_timesteps:
            # BC for policy
            self.policy.set_training_mode(True)
            self._train_bc_step(buffer=self.succ_buffer, batch_size=self.batch_size)
            self.policy.set_training_mode(False)
            
            self.num_timesteps += 1
            self._n_updates += 1

            if self.num_timesteps % log_interval == 0:
                self.logger.record("global_step", self.num_timesteps)
                self.logger.record("eval/global_step", self.num_timesteps)
                self.logger.record("train/global_step", self.num_timesteps)
                self._dump_logs()
            
            # Callback (Evaluation / Checkpointing)
            callback.update_locals(locals())
            # NOTE(gaoyuan) evaluation starts here
            if not callback.on_step():  # n_calls += 1
                break

        callback.on_training_end()
        eval_callback.eval_freq = original_eval_freq
        return self

    def learn(
        self: SelfTrajNFT,
        iterations: int, # NOTE: This refers to gradient steps in offline RL
        callback: MaybeCallback = None,
        log_interval: int = 500,
        tb_log_name: str = "traj_nft",
        reset_num_timesteps: bool = False, # False to continue from BC
        progress_bar: bool = False,
    ) -> SelfTrajNFT:
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
        print(f"\n\n{'#'*80}")
        print(f"[INFO] Starting NFT Phase for {iterations} steps...")
        print(f"[INFO] eval freq: {eval_freq}")
        print(f"[INFO] online collection freq: {self.steps_per_iter}")
        print(f"[INFO] total online collection times: {int(total_timesteps / self.steps_per_iter)}")
        print("#"*80)

        # NOTE: success buffer should be reset when start nft phase
        self.succ_buffer.reset()
        warmup_succ_buffer(
            env=self.env,
            succ_buffer=self.succ_buffer,
            num_trajectories=self.succ_warmup_traj,
            policy=self.policy,
            chunk_size= self.act_chunk,
            action_dim=self.action_dim,
            device=self.device,
            flow_steps=self.cfg.flow_steps
        )

        while self.num_timesteps < total_timesteps:
            # Collect Data (online)
            self._online_collection(self.succ_buffer, self.fail_buffer)

            self.critic.set_training_mode(False)
            self.value.train(False)
            self.policy.set_training_mode(True)

            for _ in range(self.steps_per_iter):
                self._train_nft_step(
                    succ_buffer=self.succ_buffer,
                    fail_buffer=self.fail_buffer,
                    batch_size=self.batch_size, 
                    nft_delta=self.nft_delta,
                    with_scale=self.with_scale,
                    sample_mode=self.sample_mode,
                    reward_mode=self.reward_mode,
                )

                if (self.num_timesteps + 1) % self.update_freq == 0:
                    # Polyak averaging for old policy
                    polyak_update(self.policy.parameters(), self.policy_old.parameters(), self.policy_eta)

                # Update counters
                self.num_timesteps += 1
                self._n_updates += 1

                # Logging
                if self.num_timesteps % log_interval == 0:
                    self.logger.record("global_step", self.num_timesteps)
                    self.logger.record("eval/global_step", self.num_timesteps)
                    self.logger.record("train/global_step", self.num_timesteps)
                    self.logger.record("buffer/global_step", self.num_timesteps)
                    self._dump_logs()

                # Callback (Evaluation / Checkpointing)
                callback.update_locals(locals())
                if not callback.on_step():
                    break
            
            if self.num_timesteps >= total_timesteps:
                break

            # NOTE: every time finish one loop, the fail_buffer should be reset
            self.fail_buffer.reset()

        callback.on_training_end()
        return self

    def _train_bc_step(self, buffer: ReplayBuffer, batch_size: int) -> None:
        """
        Single gradient step for Behavior Cloning, for policy only

        The input data should always be success buffer, so there is no success mask.
        """
        replay_data = buffer.sample(batch_size, env=self._vec_normalize_env)
        obs = replay_data.observations
        actions = replay_data.actions
        
        # we expected actions in shape: (B, chunk_len, act_dim)
        assert len(actions.shape) == 2, f"actions in replay buffer has no expected shape, with shape of: {actions.shape}"
        target_actions = actions.view(batch_size, self.policy.chunk_length, self.policy.act_dim).to(self.device)

        # Flow Matching Loss Logic
        x_1 = target_actions
        x_0 = torch.randn_like(x_1, device=self.device)
        t = torch.rand(batch_size, device=self.device)
        t_expand = t.view(batch_size, 1, 1) if x_1.dim() == 3 else t.view(batch_size, 1)
        
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        v_target = x_1 - x_0
        
        v_pred = self.policy(obs, x_t, t)
        loss = F.mse_loss(v_pred, v_target)

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        self.logger.record("train/policy_bc_loss", loss.item())

    def _train_nft_step(self, succ_buffer: ReplayBuffer, fail_buffer: ReplayBuffer, batch_size: int, 
                        nft_delta: float = 0.5, with_scale: bool = True, 
                        sample_mode: str = "half_succ_fail", reward_mode: str = "01") -> None:
        """
        Single gradient step for NFT.
        """
        # Determine mask values and dipole normalization parameters
        if reward_mode in ["01", "10"]:
            r_succ, r_fail = 1.0, 0.0
        elif reward_mode in ["46", "64"]:
            r_succ, r_fail = 0.6, 0.4
        else:
            raise ValueError(f"Unknown reward_mode: {reward_mode}")
        
        # sample ###########################################################################################
        n_succ = 0
        n_fail = 0
        size_succ = succ_buffer.buffer_size if succ_buffer.full else succ_buffer.pos
        size_fail = fail_buffer.buffer_size if fail_buffer.full else fail_buffer.pos

        # different sample method
        if sample_mode == "half_succ_fail":
            n_succ = batch_size // 2
            n_fail = batch_size - n_succ
        elif sample_mode == "uniform_mix":
            total_size = size_succ + size_fail
            if total_size == 0:
                raise ValueError("make sure the buffer is NOT empty!!")
            p_succ = size_succ / total_size
            n_succ = np.random.binomial(n=batch_size, p=p_succ)
            n_fail = batch_size - n_succ
        else:
            raise ValueError("unknown sample method")

        obs_list, act_list = [], []
        r_mask_list = []

        if n_succ > 0:
            if (succ_buffer.full or succ_buffer.pos > 0):
                succ_data = succ_buffer.sample(n_succ, env=self._vec_normalize_env)
                obs_list.append(succ_data.observations)
                act_list.append(succ_data.actions)
                r_mask_list.append(torch.full((n_succ,), r_succ, device=self.device))
            else:
                n_fail += n_succ
                n_succ = 0

        # Collect Fail Data
        if n_fail > 0:
            if (fail_buffer.full or fail_buffer.pos > 0):
                fail_data = fail_buffer.sample(n_fail, env=self._vec_normalize_env)
                obs_list.append(fail_data.observations)
                act_list.append(fail_data.actions)
                
                r_mask_list.append(torch.full((n_fail,), r_fail, device=self.device))
            else:
                pass

        obs = torch.cat(obs_list, dim=0)
        actions = torch.cat(act_list, dim=0)
        r_mask = torch.cat(r_mask_list, dim=0).view(batch_size, 1, 1)

        perm = torch.randperm(batch_size)
        obs = obs[perm]
        actions = actions[perm]
        r_mask = r_mask[perm]

        # training ###########################################################################################
        x_1 = actions.view(batch_size, self.policy.chunk_length, self.policy.act_dim)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(batch_size, device=self.device)
        t_expand = t.view(batch_size, 1, 1)
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1
        target_v = x_1 - x_0  # The ground truth vector field (u_t)

        with torch.no_grad():
            old_v = self.policy_old(obs, x_t, t)
        
        pred_v = self.policy(obs, x_t, t)
        delta_v_raw = target_v - old_v
        old_v_norm = torch.norm(old_v, p=2, dim=(-1, -2), keepdim=True)
        delta_v_raw_norm = torch.norm(delta_v_raw, p=2, dim=(-1, -2), keepdim=True)
        neg_norm_ratio = delta_v_raw_norm / old_v_norm * (1 - r_mask)
        
        # TODO(gaoyuan) check this
        if with_scale:
            scale_factor = torch.minimum(
                torch.tensor(1.0, device=self.device), 
                nft_delta * old_v_norm / (delta_v_raw_norm + 1e-8)
            )
        else:
            scale_factor = torch.tensor(1.0, device=self.device)
        nft_target_v = old_v + scale_factor * delta_v_raw * (2 * r_mask - 1)
        actor_loss = torch.mean((pred_v - nft_target_v) ** 2)

        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()

        # just for log
        with torch.no_grad():
            nft_beta = 1.0
            positive_v = (1.0 - nft_beta) * old_v + nft_beta * pred_v
            negative_v = (1.0 + nft_beta) * old_v - nft_beta * pred_v
            
            # regardless of whether r was 0.6 or 1.0
            loss_pos_raw = torch.mean(r_mask * (positive_v - target_v)**2)
            loss_neg_raw = torch.mean((1.0 - r_mask) * (negative_v - target_v)**2)

        # logging ###########################################################################################
        self.logger.record("train/nft_actor_loss", actor_loss.item())
        self.logger.record("train/positive_loss", loss_pos_raw.item())
        self.logger.record("train/negative_loss", loss_neg_raw.item())
        self.logger.record("train/scale_factor_mean", scale_factor.mean().item())
        self.logger.record("train/delta_v_raw_norm", delta_v_raw_norm.mean().item())
        self.logger.record("train/old_v_norm", old_v_norm.mean().item())
        self.logger.record("train/neg_norm_ratio", neg_norm_ratio.mean().item())
        self.logger.record("train/ema_eta", self.policy_eta)
        self.logger.record("buffer/success_buffer", size_succ)
        self.logger.record("buffer/fail_buffer", size_fail)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        assert self.device.type == 'cuda', f"Device Assertion Failed: Expected 'cuda', but got '{self.device}'."
        self.policy.to(self.device)

        self.policy.set_training_mode(False)
        observation, vectorized_env = self.policy.obs_to_tensor(observation)
        batch_size = observation.shape[0]

        with torch.no_grad():
            x = torch.randn(
                batch_size,
                self.policy.chunk_length, 
                self.policy.act_dim,
                device=self.device,
            )
            
            num_steps = self.cfg.flow_steps
            dt = 1.0 / num_steps
            
            for i in range(num_steps):
                t_val = i * dt
                t_tensor = torch.full((batch_size,), t_val, device=self.device)
                
                v_pred = self.policy(observation, x, t_tensor)
                x = x + v_pred * dt

            action = x.reshape(-1, self.policy.chunk_length * self.policy.act_dim)
            action = action.cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            # clip based on environment bounds, not arbitrary -1,1
            action = np.clip(action, self.action_space.low, self.action_space.high)

        return action

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        return ["policy", "critic", "value", "policy_old", "policy_optimizer", "critic_optimizer", "value_optimizer"], []


def collect_online_data(
        env: Union[GymEnv, str], 
        succ_buffer: ReplayBuffer, 
        fail_buffer: ReplayBuffer, 
        num_trajectories: int, 
        policy, 
        chunk_size: int,
        action_dim: int,
        device,
        flow_steps: int = 100,
        reward_offset: float = 1.0
    ) -> None:
    original_mode = policy.training
    policy.set_training_mode(False)
    
    env_trajectories = [
        {"obs": [], "next_obs": [], "actions": [], "rewards": [], "dones": [], "infos": []}
        for _ in range(env.num_envs)
    ]
    obs = env.reset()
    
    success_threshold = -reward_offset
    succ_trajectories = 0
    fail_trajectories = 0
    total_collected = 0
    pbar = tqdm(total=num_trajectories, desc="Collecting Trajectories")

    # NOTE: here we make sure fail data should be collected for at least one trajectory
    while fail_trajectories == 0 or total_collected < num_trajectories:
        noise = torch.randn(env.num_envs, chunk_size, action_dim, device=device)
        obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32)

        with torch.no_grad():
            x = noise
            dt = 1.0 / flow_steps
            for i in range(flow_steps):
                t_val = i * dt
                t_tensor = torch.full((env.num_envs,), t_val, device=device)
                v_pred = policy(obs_tensor, x, t_tensor)
                x = x + v_pred * dt

            action_chunk = x.cpu().numpy()
            
            if hasattr(env, "action_space") and isinstance(env.action_space, spaces.Box):
                if env.action_space.low.shape[0] == chunk_size * action_dim:
                    low = env.action_space.low.reshape(chunk_size, action_dim)
                    high = env.action_space.high.reshape(chunk_size, action_dim)
                    action_chunk = np.clip(action_chunk, low, high)
                else:
                    try:
                        action_chunk = np.clip(action_chunk, env.action_space.low, env.action_space.high)
                    except ValueError:
                        print(f"[Warning] Clip failed due to shape mismatch: Act {action_chunk.shape}, Low {env.action_space.low.shape}")

        next_obs, reward, done, info = env.step(action_chunk)

        for i in range(env.num_envs):
            traj = env_trajectories[i]
            
            real_next_obs = next_obs[i]
            if done[i] and info[i] and "terminal_observation" in info[i]:
                real_next_obs = info[i]["terminal_observation"]

            traj["obs"].append(obs[i])
            traj["next_obs"].append(real_next_obs)
            traj["actions"].append(action_chunk[i])  
            traj["rewards"].append(reward[i])
            traj["dones"].append(done[i])
            traj["infos"].append(info[i])

            if done[i]:
                is_succ = reward[i] > success_threshold
                should_save = False
                
                if is_succ:
                    if total_collected < num_trajectories or succ_trajectories == 0:
                        should_save = True
                        target_buffer = succ_buffer
                else:
                    if total_collected < num_trajectories or fail_trajectories == 0:
                        should_save = True
                        target_buffer = fail_buffer

                if should_save:
                    added = _add_traj_to_buffer(traj=traj, buffer=target_buffer)
                    if added:
                        total_collected += 1
                        if is_succ:
                            succ_trajectories += 1
                        else:
                            fail_trajectories += 1
                        
                        if total_collected <= num_trajectories:
                            pbar.update(1)
                        else:
                            pbar.set_description(f"Searching for missing {'succ' if succ_trajectories==0 else 'fail'}...")

                env_trajectories[i] = {
                    "obs": [], "next_obs": [], "actions": [],
                    "rewards": [], "dones": [], "infos": []
                }

        obs = next_obs

    pbar.close()
    policy.set_training_mode(original_mode)


def warmup_succ_buffer(
        env: Union[GymEnv, str], 
        succ_buffer: ReplayBuffer, 
        num_trajectories: int, 
        policy, 
        chunk_size: int,
        action_dim: int,
        device,
        flow_steps: int = 100,
        reward_offset: float = 1.0
    ) -> None:
    """
    Collects exactly 'num_trajectories' successful trajectories into succ_buffer.
    """
    original_mode = policy.training
    policy.set_training_mode(False)
    
    env_trajectories = [
        {"obs": [], "next_obs": [], "actions": [], "rewards": [], "dones": [], "infos": []}
        for _ in range(env.num_envs)
    ]
    obs = env.reset()
    
    success_threshold = -reward_offset
    succ_trajectories = 0
    
    pbar = tqdm(total=num_trajectories, desc="Collecting Success Trajectories")
    while succ_trajectories < num_trajectories:
        noise = torch.randn(env.num_envs, chunk_size, action_dim, device=device)
        obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32)

        with torch.no_grad():
            x = noise
            dt = 1.0 / flow_steps
            for i in range(flow_steps):
                t_val = i * dt
                t_tensor = torch.full((env.num_envs,), t_val, device=device)
                v_pred = policy(obs_tensor, x, t_tensor)
                x = x + v_pred * dt

            action_chunk = x.cpu().numpy()
            
            if hasattr(env, "action_space") and isinstance(env.action_space, spaces.Box):
                if env.action_space.low.shape[0] == chunk_size * action_dim:
                    low = env.action_space.low.reshape(chunk_size, action_dim)
                    high = env.action_space.high.reshape(chunk_size, action_dim)
                    action_chunk = np.clip(action_chunk, low, high)
                else:
                    try:
                        action_chunk = np.clip(action_chunk, env.action_space.low, env.action_space.high)
                    except ValueError:
                        print(f"[Warning] Clip failed due to shape mismatch: Act {action_chunk.shape}, Low {env.action_space.low.shape}")

        next_obs, reward, done, info = env.step(action_chunk)

        for i in range(env.num_envs):
            traj = env_trajectories[i]
            real_next_obs = next_obs[i]
            if done[i] and info[i] and "terminal_observation" in info[i]:
                real_next_obs = info[i]["terminal_observation"]

            traj["obs"].append(obs[i])
            traj["next_obs"].append(real_next_obs)
            traj["actions"].append(action_chunk[i])  
            traj["rewards"].append(reward[i])
            traj["dones"].append(done[i])
            traj["infos"].append(info[i])

            if done[i]:
                is_succ = reward[i] > success_threshold
                
                # Only save if it is successful and we haven't reached the target count yet
                if is_succ and succ_trajectories < num_trajectories:
                    added = _add_traj_to_buffer(traj=traj, buffer=succ_buffer)
                    if added:
                        succ_trajectories += 1
                        pbar.update(1)
                
                # Reset the temporary trajectory storage for this env regardless of success/fail
                env_trajectories[i] = {
                    "obs": [], "next_obs": [], "actions": [],
                    "rewards": [], "dones": [], "infos": []
                }

        obs = next_obs

    pbar.close()
    policy.set_training_mode(original_mode)
