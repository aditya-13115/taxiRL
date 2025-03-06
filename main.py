import random
import numpy as np
import gymnasium as gym  
import os
import pygame  # Import Pygame to fix rendering issues
import time

env = gym.make("Taxi-v3")

alpha = 0.9  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.9999995
min_epsilon = 0.1
num_episodes = 2000
max_steps = 100

# Define persistent directory
persist_directory = os.path.join(os.getcwd(), "q_learning_data")
os.makedirs(persist_directory, exist_ok=True)
q_table_path = os.path.join(persist_directory, "q_table.npy")

# Load existing Q-table if available, else create a new one
if os.path.exists(q_table_path):
    q_table = np.load(q_table_path)
    print("✅ Loaded existing Q-table.")
else:
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    print("🆕 No saved Q-table found. Starting fresh.")

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state, :])

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False

    for step in range(max_steps):
        action = choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Debugging print
        print(f"Episode: {episode}, Step: {step}, State: {state}, Next State: {next_state}, Done: {done}")

        if 0 <= next_state < q_table.shape[0]:  # Ensure valid state
            next_max = np.max(q_table[next_state, :])
        else:
            next_max = 0  # Default, if invalid

        old_value = q_table[state, action]
        q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        state = next_state

        if done:
            break
    epsilon = max(epsilon * epsilon_decay, min_epsilon)

print("Training completed.")

# Save the Q-table after training
np.save(q_table_path, q_table)
print("✅ Q-table saved successfully!")

# Test the trained model
env = gym.make('Taxi-v3', render_mode="human")

for episode in range(10):
    state, _ = env.reset()
    done = False

    print(f'🚖 Running Episode {episode}')

    for step in range(max_steps):
        env.render()
        pygame.event.pump()
        action = np.argmax(q_table[state, :])
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state
        done = terminated or truncated

        if done:
            env.render()
            print(f"🏁 Finished episode {episode} with reward: {reward}")
            break

env.close()
