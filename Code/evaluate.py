import numpy as np
from config import Q_TABLE_PATH, EVAL_EPISODES, SEED
from environment import make_env

def evaluate(q_values=None):
    """Evaluate a trained Q-table over multiple Taxi-v3 episodes."""
    if q_values is None:
        # Load the current q_values table
        q_values = np.load(Q_TABLE_PATH)

    # Create new environment
    env = make_env(render=False)

    # Statistics for analysing
    rewards, steps, successes, illegal = [], [], [], []
    obs, _ = env.reset(seed=SEED)

    for episode in range(EVAL_EPISODES):
        # New start
        if episode > 0:
            obs, _ = env.reset()
        step_count, illegal_count, total_reward = 0, 0, 0
        done = False

        while not done:
            # Get action (best known action using q-values)
            action = int(np.argmax(q_values[obs]))
            # Take action and observe result
            obs, reward, terminated, truncated, _ = env.step(action)
            # If violates the rule (since violate gives -10 reward), increase the count
            if reward == -10:
                illegal_count += 1

            total_reward += reward
            step_count += 1
            done = terminated or truncated 

        steps.append(step_count)
        successes.append(1 if terminated and reward == 20 else 0)
        illegal.append(illegal_count)
        rewards.append(total_reward)

    env.close()
    return {
        "avg_reward": sum(rewards) / len(rewards),
        "avg_steps": sum(steps) / len(steps),
        "success_rate": sum(successes) / len(successes) * 100,
        "avg_illegal": sum(illegal) / len(illegal)
    }