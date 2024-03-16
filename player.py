# Import necessary modules
# torch: Used for building and training neural networks.
# numpy: Used for handling arrays and numerical operations.
# PlayerNN: The neural network architecture defined in player_nn.py.
# TensorDict, TensorDictReplayBuffer, LazyMemmapStorage: Tools for efficiently managing and storing replay buffers.
import torch
import numpy as np
from player_nn import PlayerNN
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

# Define the Player class
class Player:
    def __init__(self, input_dims, num_actions, lr=0.00025, gamma=0.9, epsilon=1.0, eps_decay=0.99999975, eps_min=0.1, replay_buffer_capacity=100_000, batch_size=32, sync_network_rate=10000):
        # Store the number of possible actions
        self.num_actions = num_actions
        # Initialize the step counter for learning updates
        self.learn_step_counter = 0

        # Define player hyperparameters
        self.lr = lr  # Learning rate
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon  # Epsilon for epsilon-greedy action selection
        self.eps_decay = eps_decay  # Decay rate for epsilon
        self.eps_min = eps_min  # Minimum value for epsilon
        self.batch_size = batch_size  # Batch size for learning
        self.sync_network_rate = sync_network_rate  # How often to sync the target network with the online network

        # Initialize the online and target neural networks
        self.online_network = PlayerNN(input_dims, num_actions)
        self.target_network = PlayerNN(input_dims, num_actions, freeze=True)  # Freeze the target network to not train it directly

        # Set up the optimizer and loss function for training
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()  # Mean Squared Error Loss for the difference between predicted and target Q-values

        # Initialize the replay buffer with a specified capacity
        storage = LazyMemmapStorage(replay_buffer_capacity)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)

    # Method to decide actions based on the current state and epsilon-greedy strategy
    def choose_action(self, observation):
        # With probability epsilon, choose a random action
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        # Otherwise, choose the action with the highest Q-value predicted by the online network
        observation = torch.tensor(np.array(observation), dtype=torch.float32) \
                        .unsqueeze(0) \
                        .to(self.online_network.device)
        return self.online_network(observation).argmax().item()

    # Method to decay epsilon, ensuring exploration decreases over time
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    # Method to store an experience in the replay buffer
    def store_in_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.add(TensorDict({
                                            "state": torch.tensor(np.array(state), dtype=torch.float32),
                                            "action": torch.tensor(action),
                                            "reward": torch.tensor(reward),
                                            "next_state": torch.tensor(np.array(next_state), dtype=torch.float32),
                                            "done": torch.tensor(done)
                                          }, batch_size=[]))

    # Method to synchronize the target network with the online network at specified intervals
    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    # Methods to save and load the model
    def save_model(self, path):
        torch.save(self.online_network.state_dict(), path)

    def load_model(self, path):
        self.online_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(torch.load(path))

    # The learning method where the agent updates its knowledge
    def learn(self):
        # Wait until the replay buffer has enough samples
        if len(self.replay_buffer) < self.batch_size:
            return
        
        self.sync_networks()  # Sync target network with online network if needed
        
        self.optimizer.zero_grad()  # Reset gradients

        # Sample a batch of experiences from the replay buffer
        samples = self.replay_buffer.sample(self.batch_size).to(self.online_network.device)

        # Unpack the samples
        keys = ("state", "action", "reward", "next_state", "done")
        states, actions, rewards, next_states, dones = [samples[key] for key in keys]

        # Calculate the predicted Q-values from the online network
        predicted_q_values = self.online_network(states)
        predicted_q_values = predicted_q_values[np.arange(self.batch_size), actions.squeeze()]

        # Calculate the target Q-values from the target network
        target_q_values = self.target_network(next_states).max(dim=1)[0]
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

        # Compute the loss and perform a backward pass
        loss = self.loss(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        # Increment the learning step counter and decay epsilon
        self.learn_step_counter += 1
        self.decay_epsilon()
