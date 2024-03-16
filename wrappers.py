# Import necessary modules from NumPy and Gym.
# NumPy for numerical operations,
# Wrapper to create custom gym environment wrappers,
# and specific wrappers for observation preprocessing.
import numpy as np
from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

# Define SkipFrame, a custom wrapper to skip a fixed number of frames for each action taken.
# This is useful for speeding up the training process and making it more manageable for the agent to learn.
class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        # Initialize the parent Wrapper class with the environment.
        super().__init__(env)
        self.skip = skip  # Number of frames to skip.
    
    # Override the step function to implement frame skipping.
    def step(self, action):
        total_reward = 0.0  # Initialize the total reward for the skipped frames.
        done = False  # Initialize the done flag.
        # Repeat the action for 'skip' number of frames.
        for _ in range(self.skip):
            # Execute the action in the environment and accumulate rewards.
            next_state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:  # Break the loop if the episode ends.
                break
        # Return the result of the last frame after skipping.
        return next_state, total_reward, done, trunc, info
    

# Function to apply multiple wrappers to the environment for preprocessing observations.
def apply_wrappers(env):
    # Apply the SkipFrame wrapper to skip a fixed number of frames after each action.
    env = SkipFrame(env, skip=4)  # Skip 4 frames.
    # Resize the observation to a smaller size to reduce the computational load.
    env = ResizeObservation(env, shape=84)  # Resize to 84x84 pixels.
    # Convert the observation frames to grayscale to simplify the input for the agent.
    env = GrayScaleObservation(env)
    # Stack a specified number of consecutive frames together to provide the agent with a sense of motion.
    # This can be essential for making decisions based on the direction of movement.
    env = FrameStack(env, num_stack=4, lz4_compress=True)  # Stack 4 frames together, optionally compress to save memory.
    # Return the environment with all the wrappers applied.
    return env
