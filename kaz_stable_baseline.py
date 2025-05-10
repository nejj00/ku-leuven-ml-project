import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from utils import create_environment
from pettingzoo.utils.conversions import aec_to_parallel
import supersuit as ss
from stable_baselines3.common.vec_env import VecEnvWrapper
import numpy as np

from stable_baselines3.common.vec_env import VecEnvWrapper
import numpy as np

class VecEpisodeInfo(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self):
        self.episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return self.venv.reset()

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.episode_rewards += rewards
        self.episode_lengths += 1

        for i in range(len(dones)):
            if dones[i]:
                infos[i] = infos[i].copy()  # avoid mutating original info
                infos[i]["episode"] = {
                    "r": self.episode_rewards[i],
                    "l": self.episode_lengths[i],
                }
                self.episode_rewards[i] = 0
                self.episode_lengths[i] = 0

        return obs, rewards, dones, infos


def random_agent():
    parallel_env = create_environment(num_agents=1, render_mode="human", parrallel=True)

    observations, info = parallel_env.reset(seed=42)
    rewards = 0
    
    total_rewards = {agent: 0 for agent in parallel_env.agents}
    while parallel_env.agents:
        # this is where you would insert your policy
        actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}

        observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
        
        for key, value in rewards.items():
            total_rewards[key] += value        
        
        if any(terminations.values()) or any(truncations.values()):
            parallel_env.reset()
            break
        
    print(f"Total baseline rewards: {total_rewards}")
    parallel_env.close()

def train_agent_dqn():
    env = create_environment(num_agents=1, parrallel=True)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=2.3e-3,
        batch_size=64,
        buffer_size=100000,
        learning_starts=1000,
        gamma=0.99,
        target_update_interval=10,
        train_freq=256,
        gradient_steps=128,
        exploration_fraction=0.16,
        exploration_initial_eps=0.04,
        policy_kwargs=dict(net_arch=[256, 256]),
        
    )

    print("Starting DQN training...")
    try:
        model.learn(total_timesteps=int(1e5), log_interval=100)
    except Exception as e:
        print("Training interrupted.")
        print(e)
    finally:
        env.close()

    model.save("dqn_kaz")
    print("Training finished.")


def evaluate_agent_dqn():
    env = create_environment(num_agents=1, render_mode="human")
    model = DQN.load("dqn_kaz")

    env.reset(seed=42)
    env.action_space(env.possible_agents[0]).seed(42)

    rewards = {agent: 0 for agent in env.possible_agents}

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()

        for a in env.agents:
            rewards[a] += env.rewards[a]

        if not termination and not truncation:
            action, _ = model.predict(obs, deterministic=True)
        else:
            break
        
        env.step(action)

    print(f"Total agent rewards: {rewards}")
    env.close()
    
    avg_reward = sum(rewards.values()) / len(rewards.values())
    print(f"Avg reward: {avg_reward}")
    print("Full rewards: ", rewards)


def train_agent_ppo():
    env = create_environment(num_agents=1, parrallel=True)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")
    # env = VecEpisodeInfo(env) 
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device="cpu",
        n_steps=32,
        batch_size=256,
        gae_lambda=0.8,
        gamma=0.98,
        n_epochs=20,
        ent_coef=0.0,
        learning_rate=0.001,
        clip_range=0.2,
    )
    
    print("Starting training...")
    try:
        model.learn(total_timesteps=1e6, log_interval=100)
    except Exception as e:
        print("Training interrupted.")
        print(e)
    finally:
        env.close()
        
    model.save("ppo_kaz")
    print("Training finished.")
    
def evaluate_agent_ppo():
    env = create_environment(num_agents=1, render_mode="human")
    model = PPO.load("ppo_kaz", device="cpu")

    env.reset(seed=42)
    env.action_space(env.possible_agents[0]).seed(42)

    rewards = {agent: 0 for agent in env.possible_agents}

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()

        for a in env.agents:
            rewards[a] += env.rewards[a]

        if not termination and not truncation:
            action, _ = model.predict(obs, deterministic=True)
        else:
            break
        
        env.step(action)

    print(f"Total agent rewards: {rewards}")
    env.close()
    
    avg_reward = sum(rewards.values()) / len(rewards.values())
    print(f"Avg reward: {avg_reward}")
    print("Full rewards: ", rewards)



# train_agent_dqn()
# evaluate_agent_dqn()

# train_agent_ppo()
# evaluate_agent_ppo()

# random_agent()