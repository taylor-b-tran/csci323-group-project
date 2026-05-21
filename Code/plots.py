import matplotlib.pyplot as plt
import numpy as np
from config import RESULTS_DIR

def smooth(data, window=100):
    """Smooth a sequence using a moving average."""
    return np.convolve(data, np.ones(window)/window, mode='valid')

def save_plots(episode_rewards, episode_steps):
    """Save training reward and step plots to the results folder."""
    # Training reward curve
    plt.figure()
    plt.plot(smooth(episode_rewards), label="Smoothed Reward (window=100)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward over Episodes")
    plt.legend()
    plt.savefig(RESULTS_DIR / "training_rewards.png")
    plt.close()

    # Steps per episode curve
    plt.figure()
    plt.plot(smooth(episode_steps), label="Smoothed Steps (window=100)", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Steps per Episode during Training")
    plt.legend()
    plt.savefig(RESULTS_DIR / "training_steps.png")
    plt.close()

    print("Plots saved to results/")