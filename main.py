# Import necessary libraries for the project.
import os
import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from player import Player
from player_nn import PlayerNN  # Ensure this is imported
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
from utils import *
from torch.optim.lr_scheduler import StepLR

# Path to load the pre-trained model.
load_checkpoint_path = "/Users/cesarrivera/MarioRL/models/best_model.pt"
best_model_path = os.path.join("models", "best_model.pt")

# Setup the directory where the model and checkpoints will be saved.
model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)

# Check and use CUDA for GPU acceleration if available, otherwise use CPU.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device.upper()}")

# Configuration variables for the environment and training.
ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = True
DISPLAY = True
CKPT_SAVE_INTERVAL = 5000
NUM_OF_EPISODES = 100_000

# Initialize the game environment with the specified level and rendering mode.
env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)  # Use RIGHT_ONLY initially
env = apply_wrappers(env)

# Create an instance of the Player class, specifying the input dimensions and number of actions.
player = Player(input_dims=env.observation_space.shape, num_actions=env.action_space.n, 
                lr=0.0001, eps_decay=0.9999995, batch_size=64)

# Function to remap state dict keys
def remap_state_dict_keys(state_dict, old_prefix, new_prefix):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace(old_prefix, new_prefix)
        new_state_dict[new_key] = value
    return new_state_dict

# Load the existing model if specified
if load_checkpoint_path:
    state_dict = torch.load(load_checkpoint_path)
    state_dict = remap_state_dict_keys(state_dict, 'network.', 'network.')
    player.online_network.load_state_dict(state_dict)
    player.target_network.load_state_dict(state_dict)
    print(f"Loaded model from {load_checkpoint_path}")

# Reinitialize environment with SIMPLE_MOVEMENT after loading model
env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = apply_wrappers(env)
player.num_actions = env.action_space.n  # Update the number of actions
player.online_network = PlayerNN(input_shape=env.observation_space.shape, n_actions=player.num_actions)
player.target_network = PlayerNN(input_shape=env.observation_space.shape, n_actions=player.num_actions, freeze=True)
player.optimizer = torch.optim.Adam(player.online_network.parameters(), lr=player.lr)

# Initialize the learning rate scheduler
scheduler = StepLR(player.optimizer, step_size=10000, gamma=0.1)

# Initialize best reward to negative infinity
best_reward = float('-inf')

print("Starting training.")

# The main training loop over the specified number of episodes.
for i in range(NUM_OF_EPISODES):
    print(f"Episode: {i}")
    done = False
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).to(player.online_network.device)
    total_reward = 0
    stuck_counter = 0
    previous_state = None

    while not done:
        action = player.choose_action(state)
        new_state, reward, done, truncated, info = env.step(action)
        new_state = torch.tensor(new_state, dtype=torch.float32).to(player.online_network.device)
        total_reward += reward

        # Check if Mario is stuck
        if previous_state is not None and torch.equal(new_state, previous_state):
            stuck_counter += 1
        else:
            stuck_counter = 0
        previous_state = new_state
        
        stuck = stuck_counter > 10  # Adjust threshold as needed

        # Reward shaping
        if done and reward > 0:
            reward += 1000
        reward -= 0.1
        if stuck:
            reward -= 1

        if SHOULD_TRAIN:
            player.store_in_memory(state, action, reward, new_state, done)
            player.learn()

        state = new_state

    scheduler.step()

    if total_reward > best_reward:
        best_reward = total_reward
        player.save_model(best_model_path)
        print(f"New best model saved with total reward: {total_reward}")

    if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
        checkpoint_path = os.path.join(model_path, f"model_checkpoint_{i + 1}.pt")
        player.save_model(checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    print(f"Total reward: {total_reward}, Epsilon: {player.epsilon}")

env.close()
