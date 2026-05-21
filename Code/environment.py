import gymnasium as gym
from config import SEED

def make_env(seed=SEED, render=False):
    """Create and reset a Taxi-v3 Gym environment."""
    if render:
        env = gym.make("Taxi-v3", render_mode="human")
    else:
        env = gym.make("Taxi-v3")
    # reset environment to start a new episode
    env.reset(seed=seed)
    return env