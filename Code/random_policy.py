from config import EVAL_EPISODES
from environment import make_env

rewards, steps, successes, illegal = [], [], [], []

def run_random_policy():
    # Create environment
    env = make_env()

    for _ in range(EVAL_EPISODES):
        # Reset environment to start a new episode
        observation, info = env.reset()
        
        episode_over = False
        total_reward, step_count, illegal_count = 0, 0, 0

        while not episode_over:
            # Select random action
            action = env.action_space.sample() 

            # Take the action and see what happens
            observation, reward, terminated, truncated, info = env.step(action)

            # If violates the rule (since violate gives -10 reward), increase the count
            if reward == -10:
                illegal_count += 1

            # Get the status
            episode_over = terminated or truncated
            total_reward += reward
            step_count += 1

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