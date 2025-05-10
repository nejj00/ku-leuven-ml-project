"""
This file contains an example of implementation of the CustomWapper and CustomPredictFunction that you need to submit.

Here, we are using Ray RLLib to load the trained agents.
"""

from pathlib import Path
from typing import Optional

import gymnasium
import torch
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from ray.rllib.core.rl_module import MultiRLModule
import numpy as np

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


class CustomPredictFunction:
    """An example of an instantiation of the PredictFunction interface that loads a trained RLLib algorithm from
    a checkpoint and extracts the policies from it.
    """

    def __init__(self, env):
        # Here you should load your trained model(s) from a checkpoint in your folder
        package_directory = Path(__file__).resolve().parent
        best_checkpoint = (
            package_directory / "results" / "learner_group" / "learner" / "rl_module"
        ).resolve()

        if not best_checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {best_checkpoint}"
            )

        self.modules = MultiRLModule.from_checkpoint(best_checkpoint)

    def __call__(self, observation, agent, *args, **kwargs):
        if agent not in self.modules:
            raise ValueError(f"No policy found for agent {agent}")

        rl_module = self.modules[agent]
        fwd_ins = {"obs": torch.Tensor(observation).unsqueeze(0)}
        fwd_outputs = rl_module.forward_inference(fwd_ins)
        action_dist_class = rl_module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(fwd_outputs["action_dist_inputs"])
        action = action_dist.sample()[0].numpy()
        return action
