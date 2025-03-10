import numpy as np
import json
import os
import logging
import time

logger = logging.getLogger(__name__)

class StateManager:
    def __init__(self, state_file="snake_ai_state.json"):
        """
        Initialize the state manager with the path to the state file
        """
        self.state_file = state_file
        self.reset_lock_file = "reset_lock.tmp"
        logger.info(f"Initialized file-based StateManager with state file: {state_file}")

    def _is_reset_locked(self):
        """Check if a reset operation is in progress or recently completed"""
        try:
            if os.path.exists(self.reset_lock_file):
                with open(self.reset_lock_file, "r") as f:
                    reset_time = float(f.read().strip())
                # If reset was less than 5 seconds ago, consider it locked
                if time.time() - reset_time < 5:
                    logger.info(f"Reset lock active - {time.time() - reset_time:.2f} seconds since reset")
                    return True
                else:
                    logger.info("Reset lock expired")
                    return False
            return False
        except Exception as e:
            logger.error(f"Error checking reset lock: {e}")
            return False  # Default to not locked if there's an error reading the lock file

    def save_state(self, state_dict):
        """
        Save the current state of training to a JSON file
        """
        try:
            # Check if reset is in progress
            if self._is_reset_locked():
                logger.warning("State save prevented - reset lock is active")
                return False

            # Convert numpy values to native Python types for JSON serialization
            serializable_state = {
                'q_table': dict(state_dict['q_table']),  # Convert to normal dict
                'scores': list(state_dict['scores']),  # Convert to normal list
                'epsilon': float(state_dict['epsilon']),
                'learning_rate': float(state_dict['learning_rate'])
            }

            # Add other fields directly if they exist
            if 'steps' in state_dict:
                serializable_state['steps'] = int(state_dict['steps'])
            if 'total_reward' in state_dict:
                serializable_state['total_reward'] = float(state_dict['total_reward'])

            with open(self.state_file, 'w') as f:
                json.dump(serializable_state, f)
            logger.info(f"State saved successfully to {self.state_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving state to file: {e}")
            return False

    def load_state(self):
        """
        Load the state from file
        """
        try:
            # Check if reset is in progress
            if self._is_reset_locked():
                logger.warning("State load prevented - reset lock is active")
                return None

            # Check if file exists
            if not os.path.exists(self.state_file):
                logger.info(f"No state file found at {self.state_file}")
                return None

            with open(self.state_file, 'r') as f:
                state_dict = json.load(f)
            logger.info(f"State loaded successfully from {self.state_file}")
            return state_dict
        except Exception as e:
            logger.error(f"Error loading state from file: {e}")
            return None

    def clear_state(self):
        """Explicitly clear all saved state data"""
        try:
            # Create or update reset lock
            with open(self.reset_lock_file, "w") as f:
                f.write(str(time.time()))

            if os.path.exists(self.state_file):
                os.remove(self.state_file)
                logger.info(f"State file {self.state_file} deleted successfully")
                return True
            logger.info(f"No state file found at {self.state_file}")
            return False
        except Exception as e:
            logger.error(f"Error clearing state: {e}")
            return False