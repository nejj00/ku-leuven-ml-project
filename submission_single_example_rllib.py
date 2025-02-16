#!/usr/bin/env python3
# encoding: utf-8
"""This file contains an example of implementation of the CustomWapper and CustomPredictFunction that you need to submit.
Here, we are using Ray RLLib to load the trained agents.
You can adapt this code to your needs and remove what you do not need,
 but make sure to keep the same interface for the predict function and the custom wrapper."""

from pathlib import Path
from typing import Callable

import gymnasium
import torch
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from ray.rllib.core.rl_module import RLModule, MultiRLModule


class CustomWrapper(BaseWrapper):
    """This is an example of a custom wrapper that flattens the symbolic vector state of the environment
    Wrapper are useful to inject state pre-processing or feature that does not need to be learned by the agent

    Pay attention to submit the same (or consistent) wrapper you used during training
    """



    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return  spaces.flatten_space(super().observation_space(agent))

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        flat_obs = obs.flatten()
        return flat_obs


class CustomPredictFunction(Callable):
    """ This is an example of an instantiation of the PredictFunction interface that loads a trained RLLib algorithm from
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