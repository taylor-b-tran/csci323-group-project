import numpy as np
from config import Q_TABLE_PATH, EVAL_EPISODES
from environment import make_env

def evaluate(q_values=None):
    if q_values is None:
        # Load the current q_values table
        q_values = np.load(Q_TABLE_PATH)

    # Create new environment
    env = make_env(render=False)

    # Statistics for analysing
    rewards, steps, successes, illegal = [], [], [], []

    for _ in range(EVAL_EPISODES):
        # New start
        obs, _ = env.reset()
        total_reward, step_count, illegal_count = 0, 0, 0
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

        rewards.append(total_reward)
        steps.append(step_count)
        successes.append(1 if total_reward > 0 else 0)
        illegal.append(illegal_count)

    env.close()
    return {
        "avg_reward": sum(rewards) / len(rewards),
        "avg_steps": sum(steps) / len(steps),
        "success_rate": sum(successes) / len(successes) * 100,
        "avg_illegal": sum(illegal) / len(illegal)
    }