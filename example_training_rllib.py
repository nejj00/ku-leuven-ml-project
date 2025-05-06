#!/usr/bin/env python3
# encoding: utf-8
"""
This file contains an example of training your agents using Ray RLLib.
Notice that this code will not learn properlu, as no preprocessing of the data nor adaptation of the default PPO algorithm is done here.

"""



from pathlib import Path
from typing import Callable

import gymnasium
import pettingzoo
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module import RLModule, MultiRLModule
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
from ray.tune.registry import register_env
import numpy as np
import torch
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.dqn.torch.default_dqn_torch_rl_module import DefaultDQNTorchRLModule 
from utils import create_environment


class CustomWrapper(BaseWrapper):
    # This is an example of a custom wrapper that flattens the symbolic vector state of the environment
    # Wrapper are useful to inject state pre-processing or feature that does not need to be learned by the agent
    def __init__(self, env):
        super().__init__(env)
        self.extra_features_dim = 2  # We'll define 4 features
    
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        
        # what = spaces.flatten_space(super().observation_space(agent))

        # return  what

        base = super().observation_space(agent)
        shape = base.shape  # e.g., (N+1, 5)
        flat_obs_size = np.prod(shape)
        return spaces.Box(
            low=-1.0, high=1.0,
            shape=(flat_obs_size + self.extra_features_dim,),
            dtype=np.float64
        )


    def observe(self, agent: AgentID) -> ObsType | None:
        # obs = super().observe(agent)
        # flat_obs = obs.flatten()
        # return flat_obs
        
        obs = super().observe(agent)
        flat_obs = obs.flatten()

        # ==== Custom Features ====
        extra_feats = self._nearest_zombie_direction(obs)
        return np.concatenate([flat_obs, extra_feats])
    
    def _extract_features(self, obs_matrix):
        # obs_matrix shape: (N+1, 5) or (N+1, 11)
        # Assume zombie rows are at the end
        num_rows = obs_matrix.shape[0]
        agent_row = obs_matrix[0]
        entity_rows = obs_matrix[1:]

        # Distance is first column
        dists = entity_rows[:, 0]
        nonzero_dists = dists[dists > 0]

        # Typemasks not available? Then guess zombies from angle = [0, 1]
        # We'll assume last M rows are zombies
        num_zombies = np.sum((entity_rows[:, 4] == 1) & (entity_rows[:, 3] == 0))  # angle_y=1, angle_x=0 (crude)

        # Nearest zombie dist
        nearest_zombie_dist = np.min(nonzero_dists) if len(nonzero_dists) > 0 else 1.0

        # Average direction to zombies
        zombie_rows = entity_rows[(entity_rows[:, 4] == 1) & (entity_rows[:, 3] == 0)]
        if len(zombie_rows) > 0:
            avg_dx = np.mean(zombie_rows[:, 1])
            avg_dy = np.mean(zombie_rows[:, 2])
        else:
            avg_dx = 0.0
            avg_dy = 0.0

        return np.array([
            num_zombies / 4.0,  # Normalize by max zombies
            nearest_zombie_dist,
            avg_dx,
            avg_dy
        ], dtype=np.float64)
    
    def _nearest_zombie_direction(self, obs_matrix):
        # Assume the observation has shape (N+1, 5) or (N+1, 11)
        agent_row = obs_matrix[0]
        entity_rows = obs_matrix[1:]

        zombie_rows = entity_rows[(entity_rows[:, 4] == 1) & (entity_rows[:, 3] == 0)]

        if len(zombie_rows) == 0:
            return np.array([0.0, 0.0], dtype=np.float32)

        # Use distance column to find nearest zombie
        dists = zombie_rows[:, 0]
        idx = np.argmin(dists)
        nearest = zombie_rows[idx]

        dx = nearest[1]
        dy = nearest[2]
        norm = np.linalg.norm([dx, dy]) + 1e-8
        return np.array([dx / norm, dy / norm], dtype=np.float64)


class CustomPredictFunction(Callable):
    """ This is an example of an instantiation of the CustomPredictFunction that loads a trained RLLib algorithm from
    a checkpoint and extract the policies from it"""

    def __init__(self, env):

        # Here you should load your trained model(s) from a checkpoint in your folder
        best_checkpoint = (Path("results") / "learner_group" / "learner" / "rl_module").resolve()
        self.modules = MultiRLModule.from_checkpoint(best_checkpoint)

    def __call__(self, observation, agent, *args, **kwargs):
        rl_module = self.modules[agent]
        fwd_ins = {"obs": torch.Tensor(observation).unsqueeze(0)}
        fwd_outputs = rl_module.forward_inference(fwd_ins)
        action_dist_class = rl_module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(
            fwd_outputs["action_dist_inputs"]
        )
        action = action_dist.sample()[0].numpy()
        return action




def algo_config(id_env, policies, policies_to_train):


    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(env=id_env, disable_env_checking=True)
        .env_runners(num_env_runners=1)
        .multi_agent(
            policies={x for x in policies},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            policies_to_train=policies_to_train,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    x: RLModuleSpec(module_class=PPOTorchRLModule, model_config={"fcnet_hiddens": [128, 128]})
                    if x in policies_to_train
                    else
                    RLModuleSpec(module_class=RandomRLModule)
                    for x in policies},
            ))
        .training(
            train_batch_size=4000,
            lr=1e-4,
            gamma=0.99,
            num_sgd_iter=10,
            grad_clip=0.5,
            grad_clip_by="norm",
        )
        .debugging(log_level="ERROR")
    )

    return config


def algo_config_dqn(id_env, policies, policies_to_train):
    config = (
        DQNConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(env=id_env, disable_env_checking=True)
        .env_runners(num_env_runners=1)
        .multi_agent(
            policies={x for x in policies},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            policies_to_train=policies_to_train,
        )
        .training(
            train_batch_size=512,
            lr=1e-4,
            gamma=0.99,
        )
        .debugging(log_level="ERROR")
    )
    
    return config


def training(env, checkpoint_path, max_iterations = 500):

    # Translating the PettingZoo environment to an RLLib environment.
    # Note: RLLib use a parallelized version of the environment.
    rllib_env = ParallelPettingZooEnv(pettingzoo.utils.conversions.aec_to_parallel(env))
    id_env = "knights_archers_zombies_v10"
    register_env(id_env, lambda config: rllib_env)

    # Fix seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Define the configuration for the PPO algorithm
    policies = [x for x in env.agents]
    policies_to_train = policies
    config = algo_config(id_env, policies, policies_to_train)

    # Train the model
    algo = config.build()
    for i in range(max_iterations):
        result = algo.train()
        result.pop("config")
        if "env_runners" in result and "agent_episode_returns_mean" in result["env_runners"]:
            print(i, result["env_runners"]["agent_episode_returns_mean"])
            if result["env_runners"]["agent_episode_returns_mean"]["archer_0"] > 50: # Or any early stopping criterion
                break
        if i % 5 == 0:
            save_result = algo.save(checkpoint_path)
            path_to_checkpoint = save_result.checkpoint.path
            print(
                "An Algorithm checkpoint has been created inside directory: "
                f"'{path_to_checkpoint}'."
            )



if __name__ == "__main__":

    num_agents = 1
    visual_observation = False

    # Create the PettingZoo environment for training
    env = create_environment(num_agents=num_agents, visual_observation=visual_observation)
    env = CustomWrapper(env)

    # Running training routine
    checkpoint_path = str(Path("results").resolve())
    training(env, checkpoint_path, max_iterations = 500)
