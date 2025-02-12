from pathlib import Path

import gymnasium
import torch
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from ray.rllib.core.rl_module import RLModule, MultiRLModule


from utils import PredictFunction, create_environment



class CustomWrapper(BaseWrapper):
    """Wrapper used to do state pre-processing on the symbolic vector
    representation (e.g. feature engineering) that does not need to be
    learned by the agent

    Use this to change the representation for your learner such that it is
    consistently applied for both learning and evaluation.
    """

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        # This is an example of a custom wrapper that flattens the symbolic
        # vector state of the environment
        return  spaces.flatten_space(super().observation_space(agent))

    def observe(self, agent: AgentID) -> ObsType | None:
        # This is an example of a custom wrapper that flattens the symbolic
        # vector state of the environment
        obs = super().observe(agent)
        flat_obs = obs.flatten()
        return flat_obs


class CustomPredictFunction(PredictFunction):
    """A custom PredictFunction interface that you can adapt to call your
    trained model and return the action you want to play.
    """

    def __init__(self, env):
        # Here you should load your trained model(s) from a checkpoint in your folder
        best_checkpoint = (Path("results") / "learner_group" / "learner" / "rl_module").resolve()
        self.modules = MultiRLModule.from_checkpoint(best_checkpoint)

    def __call__(self, observation, agent, *args, **kwargs):
        # This is an example of an instantiation of the PredictFunction
        # interface that loads a trained RLLib algorithm from
        # a checkpoint and extract the policies from it
        rl_module = self.modules[agent]
        fwd_ins = {"obs": torch.Tensor(observation).unsqueeze(0)}
        fwd_outputs = rl_module.forward_inference(fwd_ins)
        action_dist_class = rl_module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(
            fwd_outputs["action_dist_inputs"]
        )
        action = action_dist.sample()[0].numpy()
        return action

