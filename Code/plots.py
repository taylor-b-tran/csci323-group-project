import matplotlib.pyplot as plt
import numpy as np

def smooth(data, window=100):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def save_plots(episode_rewards, episode_steps):
    # Training reward curve
    plt.figure()
    plt.plot(smooth(episode_rewards), label="Smoothed Reward (window=100)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward over Episodes")
    plt.legend()
    plt.savefig("results/training_rewards.png")
    plt.close()

    # Steps per episode curve
    plt.figure()
    plt.plot(smooth(episode_steps), label="Smoothed Steps (window=100)", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Steps per Episode during Training")
    plt.legend()
    plt.savefig("results/training_steps.png")
    plt.close()

    print("Plots saved to results/")