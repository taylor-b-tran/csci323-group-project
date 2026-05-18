from tqdm import tqdm
from config import Q_TABLE_PATH, EPISODES, EPSILON_DECAY, EPSILON_MIN, EPSILON_START, ALPHA, GAMMA
from environment import make_env
import numpy as np

def training():
    env = make_env()

    # Q-table: maps (state, action) to expected reward
    q_values = np.zeros((env.observation_space.n, env.action_space.n))
    
    episode_rewards = []
    episode_steps = []
    epsilion = EPSILON_START
    
    for episode in tqdm(range(EPISODES)):
        # New start
        obs, info = env.reset()
        done = False
        total_reward, steps = 0, 0

        while not done:
            # Get action
            # If random action 
            if np.random.random() < epsilion:
                action = env.action_space.sample()
            # If exploit (best known action)
            else:
                action = int(np.argmax(q_values[obs]))
            
            # Take action and observe result
            next_obs, reward, terminated, truncated, ino = env.step(action)
            done = terminated or truncated
            
            # Q-Learning update: Q(s, a) = Q(s, a) + alpha * [r + discount_factor * max Q(s', a') - Q(s, a)]
            # Calculate max Q(s', a') - best_next
            best_next = np.max(q_values[next_obs])
            q_values[obs, action] += ALPHA * (reward + GAMMA * best_next - q_values[obs, action])

            obs = next_obs
            total_reward += reward
            steps += 1        

        # After finishing one training phase, update the epsilion to gradually increase the 
        # rate of choosing optimal strategy instead of guessing
        epsilion = max(EPSILON_MIN, epsilion * EPSILON_DECAY)

        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode+1}/{EPISODES} | Avg Reward: {sum(episode_rewards[-1000:])/1000:.2f} | Epsilon: {epsilion:.3f}")

    env.close()
    # Store q_value tables to use for evaluation pipeline
    np.save(Q_TABLE_PATH, q_values)

    print(f"Q-table saved to {Q_TABLE_PATH}")
    return q_values, episode_rewards, episode_steps