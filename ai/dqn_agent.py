import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from game.constants import *

class DQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQNetwork, self).__init__()

        # Separate paths for danger/direction and food/obstacle information
        self.danger_path = nn.Sequential(
            nn.Linear(7, hidden_size),  # 3 danger + 4 direction
            nn.ReLU6(),
            nn.BatchNorm1d(hidden_size)
        )

        self.food_path = nn.Sequential(
            nn.Linear(8, hidden_size),  # 4 food + 4 obstacle
            nn.ReLU6(),
            nn.BatchNorm1d(hidden_size)
        )

        self.combined = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU6(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # Add batch dimension if input is a single state
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # Split input into two paths
        danger_x = x[:, :7]  # First 7 inputs (danger + direction)
        food_x = x[:, 7:]   # Last 8 inputs (food + obstacle)

        # Process each path
        danger_features = self.danger_path(danger_x)
        food_features = self.food_path(food_x)

        # Combine features
        combined = torch.cat((danger_features, food_features), dim=1)
        return self.combined(combined)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.memory = ReplayBuffer(10000)  # Experience replay buffer
        self.batch_size = 32  # Reduced from 64 for faster updates
        self.gamma = DISCOUNT_FACTOR
        self.epsilon = EPSILON
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = LEARNING_RATE
        self.target_update = 5  # Update target network more frequently
        self.hidden_size = 64  # Reduced from 128 for simpler architecture

        # Networks
        self.policy_net = DQNetwork(state_size, self.hidden_size, action_size).to(self.device)
        self.target_net = DQNetwork(state_size, self.hidden_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # Training monitoring
        self.total_steps = 0
        self.episode_reward = 0
        print(f"DQN Agent initialized - Device: {self.device}", flush=True)

    def get_action(self, state):
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            q_values = self.policy_net(state_tensor)
            print(f"\nQ-values: {q_values.cpu().numpy()}", flush=True)
            return q_values.argmax().item()

    def train(self, state, action, reward, next_state, done):
        # Store transition in replay memory
        self.memory.push(state, action, reward, next_state, done)
        self.total_steps += 1
        self.episode_reward += reward

        # Start training only when we have enough samples
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch from memory
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))

        # Prepare batch
        state_batch = torch.FloatTensor(np.array(batch[0])).to(self.device)
        action_batch = torch.LongTensor(batch[1]).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).to(self.device)

        # Current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # Compute loss and optimize
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Print training progress periodically
        if self.total_steps % 100 == 0:
            print(f"\nTraining Stats (Step {self.total_steps}):", flush=True)
            print(f"Episode Reward: {self.episode_reward:.2f}", flush=True)
            print(f"Epsilon: {self.epsilon:.3f}", flush=True)
            print(f"Loss: {loss.item():.4f}", flush=True)
            print(f"Average Q-value: {current_q_values.mean().item():.4f}", flush=True)
            print("-" * 40, flush=True)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print("\nTarget network updated", flush=True)

    def save(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        return path

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']