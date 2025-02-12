import abc
from typing import Callable

import gymnasium
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
import supersuit as ss
from pettingzoo.butterfly import knights_archers_zombies_v10


MAX_ZOMBIES = 4


class CustomWrapper(BaseWrapper):
    # This is an example of a custom wrapper that flattens the symbolic vector state of the environment
    # Wrapper are useful to inject state pre-processing or feature that does not need to be learned by the agent

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return  spaces.flatten_space(super().observation_space(agent))

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        flat_obs = obs.flatten()
        return flat_obs



class PredictFunction(Callable, abc.ABC):
    """A callable abstract class that takes an observation and an agent and returns an action.
    You must implement your function. In the constructor of this class you can load your model and
    implement any custom logic. Make sure that this is coherent with the training logic."""

    @abc.abstractmethod
    def __call__(self, observation, agent, *args, **kwargs):
        pass



def create_environment(num_agents=1, max_cycles = 1000, render_mode = None, visual_observation = False):
    """Create the PettingZoo environment """
    assert num_agents > 0 and num_agents < 3, "Number of agents must be either 1 (one archer) or 2 (two archers)"

    env = knights_archers_zombies_v10.env(max_cycles=max_cycles,
                                          num_archers=num_agents,
                                          num_knights= 0,
                                          max_zombies=MAX_ZOMBIES,
                                          vector_state=not visual_observation,
                                          render_mode = render_mode)
    env = ss.black_death_v3(env)
    if visual_observation:
        # If the observation space is visual, reduce the color channels, resize from 512px to 84px, and apply frame stacking
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    return env


