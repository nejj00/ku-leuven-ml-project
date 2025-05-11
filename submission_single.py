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
        self.extra_features_dim = 10  # We'll define 4 features
    
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
        extra_feats = self._compute_features(obs)
        return np.concatenate([flat_obs, extra_feats])
    
    def _compute_features(self, obs_matrix):
        agent_row = obs_matrix[0]
        entity_rows = obs_matrix[1:]

        # === Heading vector ===
        heading_x, heading_y = agent_row[3], agent_row[4]

        # === Find zombies ===
        zombies = entity_rows[(entity_rows[:, 3] == 0.0) & (entity_rows[:, 4] == 1.0)]
        num_zombies = len(zombies)

        if num_zombies == 0:
            avg_dist = 1.0
            nearest_dist = 1.0
            avg_dx, avg_dy = 0.0, 0.0
            nearest_dx, nearest_dy = 0.0, 0.0
            dot_heading_enemy_dir = 0.0
            nearest_end_dx, nearest_end_dy = 0.0, 0.0
        else:
            dists = zombies[:, 0]
            dxs = zombies[:, 1]
            dys = zombies[:, 2]

            avg_dist = np.mean(dists)
            nearest_idx = np.argmin(dists)
            nearest_dist = dists[nearest_idx]
            nearest_dx, nearest_dy = dxs[nearest_idx], dys[nearest_idx]

            avg_dx = np.mean(dxs)
            avg_dy = np.mean(dys)

            avg_dir_norm = np.linalg.norm([avg_dx, avg_dy]) + 1e-8
            avg_dir_unit = np.array([avg_dx / avg_dir_norm, avg_dy / avg_dir_norm])
            dot_heading_enemy_dir = heading_x * avg_dir_unit[0] + heading_y * avg_dir_unit[1]
            
            nearest_end_idx = np.argmax(dys)
            nearest_end_dx, nearest_end_dy = dxs[nearest_end_idx], dys[nearest_end_idx]

        # Normalize nearest zombie direction
        norm = np.linalg.norm([nearest_dx, nearest_dy]) + 1e-8
        nearest_dxdy = np.array([nearest_dx / norm, nearest_dy / norm])

        features = np.array([
            heading_x, heading_y,
            num_zombies / 4.0,  # Normalize by max zombies (assumed max = 5)
            avg_dist,
            nearest_dist,
            *nearest_dxdy,
            dot_heading_enemy_dir,
            nearest_end_dx, nearest_end_dy
        ], dtype=np.float64)

        return features


class CustomPredictFunction:
    """An example of an instantiation of the PredictFunction interface that loads a trained RLLib algorithm from
    a checkpoint and extracts the policies from it.
    """

    def __init__(self, env):
        # Here you should load your trained model(s) from a checkpoint in your folder
        package_directory = Path(__file__).resolve().parent
        best_checkpoint = (
            package_directory / "models" / "single_agent" / "learner_group" / "learner" / "rl_module"
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