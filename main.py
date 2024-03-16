# Import necessary libraries for the project.
# os: To interact with the file system.
# torch: The main PyTorch library for deep learning.
# gym_super_mario_bros: Environment library for Super Mario Bros.
# RIGHT_ONLY: A predefined action set for the Mario player.
# Player: The custom player class for handling actions, learning, etc.
# JoypadSpace: A wrapper to simplify the action space.
# apply_wrappers: Function to apply custom wrappers to the environment for preprocessing.
# utils: Contains utility functions, like getting the current date-time string.
import os
import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from player import Player
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
from utils import *

# Define a path for loading a pre-trained model. Set to None to start from scratch.
# This allows for continued training from a previous state or evaluation of a trained model.
load_checkpoint_path = None  # Example: 'models/model_checkpoint_5000.pt'

# Setup the directory where the model and checkpoints will be saved.
# It uses the current date and time to create a unique directory for this session.
model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)  # Creates the directory if it doesn't exist.

# Check and use CUDA for GPU acceleration if available, otherwise use CPU.
# This significantly speeds up training time by utilizing GPU hardware.
if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")

# Configuration variables for the environment and training.
# ENV_NAME: Identifies which level of Super Mario Bros to play.
# SHOULD_TRAIN: Flag to toggle training mode.
# DISPLAY: Controls whether the game is rendered visually during training.
# CKPT_SAVE_INTERVAL: How often to save a checkpoint of the model.
# NUM_OF_EPISODES: The total number of episodes to run during training.
ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = True
DISPLAY = True
CKPT_SAVE_INTERVAL = 5000
NUM_OF_EPISODES = 50_000

# Initialize the game environment with the specified level and rendering mode.
# The environment is wrapped to limit actions to right-only movements and apply preprocessing.
env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env)

# Create an instance of the Player class, specifying the input dimensions and number of actions.
# This player will interact with the environment, making decisions, and learning from them.
player = Player(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

# Attempt to load a pre-trained model if a checkpoint path is provided and the file exists.
# This allows for continuation of training or evaluation without starting from scratch.
if load_checkpoint_path and os.path.exists(load_checkpoint_path):
    player.load_model(load_checkpoint_path)
    print(f"Model loaded from {load_checkpoint_path}")
else:
    print("No checkpoint found, starting training from scratch.")

# The main training loop over the specified number of episodes.
for i in range(NUM_OF_EPISODES):
    print("Episode:", i)
    done = False  # Flag to indicate when the episode ends.
    state, _ = env.reset()  # Reset the environment to start a new episode, get initial state.
    total_reward = 0  # Track the total reward accumulated in the episode.

    # Loop until the episode is finished.
    while not done:
        action = player.choose_action(state)  # Player chooses an action based on the current state.
        # Execute the action in the environment, obtain new state and reward.
        new_state, reward, done, truncated, info = env.step(action)
        total_reward += reward  # Update the total reward.

        # If training mode is enabled, store the experience and perform a learning step.
        if SHOULD_TRAIN:
            player.store_in_memory(state, action, reward, new_state, done)
            player.learn()

        state = new_state  # Update the current state to the new state.

    # Save a model checkpoint at specified intervals if in training mode.
    if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
        checkpoint_path = os.path.join(model_path, f"model_checkpoint_{i + 1}.pt")
        player.save_model(checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    # Print a summary of the episode including the total reward and the current epsilon value.
    print("Total reward:", total_reward, "Epsilon:", player.epsilon)

env.close()  # Ensure the environment is closed properly when training is finished.