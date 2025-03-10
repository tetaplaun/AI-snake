import numpy as np
from game.constants import *

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.learning_rate = LEARNING_RATE
        self.gamma = DISCOUNT_FACTOR
        self.epsilon = EPSILON
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995

    def _get_state_key(self, state):
        return tuple(state.astype(int))

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)

        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)

        return np.argmax(self.q_table[state_key])

    def train(self, state, action, reward, next_state, done):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        current_q = self.q_table[state_key][action]
        if done:
            next_max_q = 0
        else:
            next_max_q = np.max(self.q_table[next_state_key])

        new_q = current_q + self.learning_rate * (
            reward + self.gamma * next_max_q - current_q)
        self.q_table[state_key][action] = new_q

        # Decay epsilon
        if not done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def load_state(self, saved_state):
        if saved_state and 'q_table' in saved_state:
            self.q_table = saved_state['q_table']
            self.epsilon = saved_state.get('epsilon', self.epsilon)
            print(f"Loaded Q-table with {len(self.q_table)} states and epsilon: {self.epsilon}", flush=True)