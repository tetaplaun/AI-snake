import json
import numpy as np
import os

class StateManager:
    def __init__(self):
        self.save_file = "snake_ai_state.json"

    def save_state(self, state_dict):
        """
        Save the current state of training
        """
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_state = {
                'q_table': {str(k): v.tolist() for k, v in state_dict['q_table'].items()},
                'scores': state_dict['scores']
            }
            
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
            
            # Convert lists back to numpy arrays
            q_table = {eval(k): np.array(v) for k, v in state_dict['q_table'].items()}
            
            return {
                'q_table': q_table,
                'scores': state_dict['scores']
            }
        except Exception as e:
            print(f"Error loading state: {e}")
            return None
