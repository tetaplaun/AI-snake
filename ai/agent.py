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
        self.last_scores = []

        # Adaptive learning rate parameters
        self.min_learning_rate = 0.0001
        self.max_learning_rate = 0.01
        self.performance_window = 10
        self.learning_rate_increase = 1.05
        self.learning_rate_decrease = 0.95

    def _get_state_key(self, state):
        return tuple(state.astype(int))

    def get_action(self, state):
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)

        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)

        # Add small random noise to break ties
        return np.argmax(self.q_table[state_key] + np.random.uniform(0, 0.01, self.action_size))

    def adjust_learning_rate(self, score):
        """Adjust learning rate based on recent performance"""
        self.last_scores.append(score)
        if len(self.last_scores) > self.performance_window:
            self.last_scores.pop(0)

            # Calculate performance trend
            if len(self.last_scores) >= 2:
                recent_avg = np.mean(self.last_scores[-5:])
                previous_avg = np.mean(self.last_scores[:-5])

                # Adjust learning rate based on performance
                if recent_avg > previous_avg:
                    # If improving, increase learning rate
                    self.learning_rate = min(
                        self.max_learning_rate,
                        self.learning_rate * self.learning_rate_increase
                    )
                else:
                    # If not improving, decrease learning rate
                    self.learning_rate = max(
                        self.min_learning_rate,
                        self.learning_rate * self.learning_rate_decrease
                    )

    def train(self, state, action, reward, next_state, done, score=0):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)

        # Initialize Q-values if not exist
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        # Q-learning update
        current_q = self.q_table[state_key][action]
        if done:
            next_max_q = 0
            # Adjust learning rate at episode end
            self.adjust_learning_rate(score)
        else:
            next_max_q = np.max(self.q_table[next_state_key])

        # Update Q-value with experience replay
        new_q = current_q + self.learning_rate * (
            reward + self.gamma * next_max_q - current_q)
        self.q_table[state_key][action] = new_q

        # Adaptive epsilon decay based on recent performance
        if not done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def load_state(self, saved_state):
        """Load agent state from a saved state dictionary"""
        try:
            # Load the Q-table
            if 'q_table' in saved_state:
                # Convert any string keys back to proper format
                q_table = {}
                for state_key_str, q_values in saved_state['q_table'].items():
                    # If the key is stored as a string representation of a tuple, convert it back
                    try:
                        if state_key_str.startswith('(') and state_key_str.endswith(')'):
                            state_key = eval(state_key_str)
                        else:
                            state_key = state_key_str
                    except:
                        state_key = state_key_str

                    # Convert values to numpy array if they're a list
                    if isinstance(q_values, list):
                        q_table[state_key] = np.array(q_values)
                    else:
                        q_table[state_key] = q_values

                self.q_table = q_table

            # Load epsilon and learning rate if available
            self.epsilon = saved_state.get('epsilon', self.epsilon)
            self.learning_rate = saved_state.get('learning_rate', self.learning_rate)
            print(f"Loaded Q-table with {len(self.q_table)} states, epsilon: {self.epsilon}, learning rate: {self.learning_rate}", flush=True)
            return True
        except Exception as e:
            print(f"Error loading agent state: {e}", flush=True)
            return False

    def reset(self):
        """
        Reset the agent to initial state.
        This clears all learning history and sets parameters back to default values.
        """
        # Clear the Q-table and reset all parameters to initial values
        self.q_table = {}
        self.epsilon = 0.1
        self.learning_rate = 0.001
        self.gamma = 0.9
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.01
        self.learning_rate_min = 0.0001
        self.min_exploration_episodes = 1000

        print("Agent reset to initial state with empty Q-table", flush=True)
        return True