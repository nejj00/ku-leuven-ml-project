import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from utils import create_environment
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1
from pettingzoo.utils.conversions import aec_to_parallel
import supersuit as ss


def random_agent():
    env = create_environment(num_agents=1, render_mode="human")

    env.reset(seed=None)
    rewards = 0

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        action = env.action_space(agent).sample() if not termination and not truncation else None
        env.step(action)
        rewards += reward

    print(f"Total baseline rewards: {rewards}")
    env.close()

def train_agent():
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


def evaluate_agent():
    env = create_environment(num_agents=1, render_mode="human")
    model = DQN.load("dqn_kaz")

    env.reset(seed=None)
    rewards = 0

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        if not termination and not truncation:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = None
        env.step(action)
        rewards += reward

    print(f"Total agent rewards: {rewards}")
    env.close()


# train_agent()
evaluate_agent()
# random_agent()