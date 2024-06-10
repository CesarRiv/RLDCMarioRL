# Import necessary libraries for the project.
import os
import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from player import Player
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
from utils import *
from torch.optim.lr_scheduler import StepLR

# Define a path for loading a pre-trained model. Set to None to start from scratch.
load_checkpoint_path = None  # Set to None to start from scratch
best_model_path = os.path.join("models", "best_model.pt")  # Path to save the best model

# Setup the directory where the model and checkpoints will be saved.
model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)  # Creates the directory if it doesn't exist.

# Check and use CUDA for GPU acceleration if available, otherwise use CPU.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device.upper()}")

# Configuration variables for the environment and training.
ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = True
DISPLAY = False  # Set to False to reduce overhead from rendering
CKPT_SAVE_INTERVAL = 5000
NUM_OF_EPISODES = 100_000  # Increase the number of episodes

# Initialize the game environment with the specified level and rendering mode.
env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

# Create an instance of the Player class, specifying the input dimensions and number of actions.
player = Player(input_dims=env.observation_space.shape, num_actions=env.action_space.n, 
                lr=0.0001, eps_decay=0.9999995, batch_size=64)  # Adjusted hyperparameters

# Initialize the learning rate scheduler
scheduler = StepLR(player.optimizer, step_size=10000, gamma=0.1)

# Initialize best reward to negative infinity
best_reward = float('-inf')

print("Starting training from scratch.")

# The main training loop over the specified number of episodes.
for i in range(NUM_OF_EPISODES):
    print("Episode:", i)
    done = False  # Flag to indicate when the episode ends.
    state, _ = env.reset()  # Reset the environment to start a new episode, get initial state.
    state = torch.tensor(state, dtype=torch.float32).to(player.online_network.device)
    total_reward = 0  # Track the total reward accumulated in the episode.

    # Loop until the episode is finished.
    while not done:
        action = player.choose_action(state)  # Player chooses an action based on the current state.
        # Execute the action in the environment, obtain new state and reward.
        new_state, reward, done, truncated, info = env.step(action)
        new_state = torch.tensor(new_state, dtype=torch.float32).to(player.online_network.device)
        total_reward += reward  # Update the total reward.

        # Reward shaping for faster completion
        if done and reward > 0:
            reward += 1000  # Bonus for level completion
        reward -= 0.1  # Small penalty to incentivize faster completion

        # If training mode is enabled, store the experience and perform a learning step.
        if SHOULD_TRAIN:
            player.store_in_memory(state, action, reward, new_state, done)
            player.learn()

        state = new_state  # Update the current state to the new state.

    # Adjust the learning rate
    scheduler.step()

    # Check if the current model is the best based on total reward
    if total_reward > best_reward:
        best_reward = total_reward
        player.save_model(best_model_path)
        print(f"New best model saved with total reward: {total_reward}")

    # Save a model checkpoint at specified intervals if in training mode.
    if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
        checkpoint_path = os.path.join(model_path, f"model_checkpoint_{i + 1}.pt")
        player.save_model(checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    # Print a summary of the episode including the total reward and the current epsilon value.
    print("Total reward:", total_reward, "Epsilon:", player.epsilon)

env.close()  # Ensure the environment is closed properly when training is finished.
