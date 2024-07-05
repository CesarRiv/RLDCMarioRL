# Import necessary modules
import torch
import numpy as np
from player_nn import PlayerNN
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class Player:
    def __init__(self, input_dims, num_actions, lr=0.00025, gamma=0.9, epsilon=1.0, eps_decay=0.99999975, eps_min=0.1, replay_buffer_capacity=100_000, batch_size=32, sync_network_rate=10000):
        self.num_actions = num_actions
        self.learn_step_counter = 0

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.sync_network_rate = sync_network_rate

        self.online_network = PlayerNN(input_dims, num_actions)
        self.target_network = PlayerNN(input_dims, num_actions, freeze=True)

        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()

        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        observation = torch.tensor(np.array(observation), dtype=torch.float32).unsqueeze(0).to(self.online_network.device)
        return self.online_network(observation).argmax().item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def store_in_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    def save_model(self, path):
        torch.save(self.online_network.state_dict(), path)

    def load_model(self, path):
        state_dict = torch.load(path)
        state_dict = self.remap_state_dict_keys(state_dict, 'network.', 'fc_layers.')
        self.online_network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict)

    def remap_state_dict_keys(self, state_dict, old_prefix, new_prefix):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace(old_prefix, new_prefix)
            new_state_dict[new_key] = value
        return new_state_dict

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        self.sync_networks()

        self.optimizer.zero_grad()

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(self.online_network.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.online_network.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.online_network.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.online_network.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.online_network.device)

        predicted_q_values = self.online_network(states)
        predicted_q_values = predicted_q_values[np.arange(self.batch_size), actions]

        target_q_values = self.target_network(next_states).max(dim=1)[0]
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones)

        loss = self.loss(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_epsilon()
