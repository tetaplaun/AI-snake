import json
import numpy as np
import os
import torch

class StateManager:
    def __init__(self):
        self.save_file = "snake_ai_state.json"
        self.model_file = "snake_dqn_model.pth"

    def save_state(self, state_dict):
        """
        Save the current state of training
        """
        try:
            # Handle both Q-learning and DQN states
            if 'q_table' in state_dict:
                # Q-learning state
                serializable_state = {
                    'q_table': {str(k): v.tolist() for k, v in state_dict['q_table'].items()},
                    'scores': state_dict['scores'],
                    'agent_type': 'q_learning'
                }

                with open(self.save_file, 'w') as f:
                    json.dump(serializable_state, f)
            else:
                # DQN state
                torch.save(state_dict['model_state'], self.model_file)
                with open(self.save_file, 'w') as f:
                    json.dump({
                        'scores': state_dict['scores'],
                        'agent_type': 'dqn'
                    }, f)
            return True
        except Exception as e:
            print(f"Error saving state: {e}", flush=True)
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

            if state_dict.get('agent_type') == 'q_learning':
                # Load Q-learning state
                q_table = {eval(k): np.array(v) for k, v in state_dict['q_table'].items()}
                return {
                    'q_table': q_table,
                    'scores': state_dict['scores'],
                    'agent_type': 'q_learning'
                }
            else:
                # Load DQN state
                if os.path.exists(self.model_file):
                    model_state = torch.load(self.model_file)
                    return {
                        'model_state': model_state,
                        'scores': state_dict['scores'],
                        'agent_type': 'dqn'
                    }
                else:
                    print("Model file not found, starting fresh", flush=True)
                    return None
        except Exception as e:
            print(f"Error loading state: {e}", flush=True)
            return None