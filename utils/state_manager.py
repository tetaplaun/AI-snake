import json
import numpy as np
import os
import torch

class StateManager:
    def __init__(self):
        self.save_file = "snake_ai_state.json"
        self.model_file = "dqn_model.pth"

    def save_state(self, state_dict):
        """
        Save the current state of training
        """
        try:
            # Save DQN model if present
            if 'dqn_agent' in state_dict:
                state_dict['dqn_agent'].save(self.model_file)

            # Convert numpy arrays to lists for JSON serialization
            serializable_state = {
                'scores': state_dict['scores'],
                'agent_type': state_dict.get('agent_type', 'q_learning')
            }

            # Add Q-table if using Q-learning
            if 'q_table' in state_dict:
                serializable_state['q_table'] = {str(k): v.tolist() for k, v in state_dict['q_table'].items()}

            with open(self.save_file, 'w') as f:
                json.dump(serializable_state, f)
            return True
        except Exception as e:
            print(f"Error saving state: {e}")
            return False

    def load_state(self):
        """
        Load the previous state of training
        """
        try:
            if not os.path.exists(self.save_file):
                return None

            with open(self.save_file, 'r') as f:
                state_dict = json.load(f)

            # Convert lists back to numpy arrays for Q-learning
            if 'q_table' in state_dict:
                q_table = {eval(k): np.array(v) for k, v in state_dict['q_table'].items()}
                state_dict['q_table'] = q_table

            return state_dict
        except Exception as e:
            print(f"Error loading state: {e}")
            return None