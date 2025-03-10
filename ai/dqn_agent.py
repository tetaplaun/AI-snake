import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os # Added import statement
from game.constants import *

class DQNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}, State size: {state_size}", flush=True)  # Debug log

        # DQN hyperparameters
        self.memory_size = 10000
        self.batch_size = 32
        self.gamma = DISCOUNT_FACTOR
        self.epsilon = EPSILON
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = LEARNING_RATE
        self.target_update = 10  # Update target network every N episodes
        self.hidden_size = 128

        # Initialize networks and memory
        self.policy_net = DQNNetwork(state_size, self.hidden_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, self.hidden_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayMemory(self.memory_size)

        self.episode_count = 0

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            print(f"State tensor shape: {state_tensor.shape}", flush=True)  # Debug log
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def train(self, state, action, reward, next_state, done):
        # Store transition in memory
        self.memory.push(state, action, reward, next_state, done)

        # Start training only when we have enough samples
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch from memory
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))

        # Prepare batch
        state_batch = torch.FloatTensor(np.array(batch[0])).to(self.device)
        action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).unsqueeze(1).to(self.device)

        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)

        # Compute target Q values
        target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # Compute loss and update policy network
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network
        if self.episode_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if done:
            self.episode_count += 1

    def save(self, path='dqn_model.pth'):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count
        }, path)

    def load(self, path='dqn_model.pth'):
        if not os.path.exists(path):
            return False

        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']
        return True