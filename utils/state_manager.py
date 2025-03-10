import numpy as np
from models import db, QTableEntry, TrainingMetrics
from web.app import app

class StateManager:
    def __init__(self):
        # Ensure database tables exist
        with app.app_context():
            db.create_all()

    def save_state(self, state_dict):
        """
        Save the current state of training to PostgreSQL
        """
        try:
            with app.app_context():
                # Save Q-table entries
                for state_key, q_values in state_dict['q_table'].items():
                    state_key_str = str(state_key)
                    for action, q_value in enumerate(q_values):
                        entry = QTableEntry.query.filter_by(
                            state_key=state_key_str,
                            action=action
                        ).first()

                        if entry:
                            entry.q_value = float(q_value)
                        else:
                            entry = QTableEntry(
                                state_key=state_key_str,
                                action=action,
                                q_value=float(q_value)
                            )
                            db.session.add(entry)

                # Save training metrics
                metrics = TrainingMetrics(
                    episode=len(state_dict['scores']),
                    score=state_dict['scores'][-1],
                    steps=state_dict.get('steps', 0),
                    total_reward=state_dict.get('total_reward', 0.0),
                    epsilon=state_dict.get('epsilon', 0.1)
                )
                db.session.add(metrics)

                db.session.commit()
            return True
        except Exception as e:
            print(f"Error saving state to database: {e}", flush=True)
            db.session.rollback()
            return False

    def load_state(self):
        """
        Load the previous state of training from PostgreSQL
        """
        try:
            with app.app_context():
                # Load Q-table entries
                q_table = {}
                entries = QTableEntry.query.all()

                for entry in entries:
                    state_key = eval(entry.state_key)  # Convert string back to tuple
                    if state_key not in q_table:
                        q_table[state_key] = np.zeros(3)  # Initialize with number of actions
                    q_table[state_key][entry.action] = entry.q_value

                # Load latest metrics
                latest_metrics = TrainingMetrics.query.order_by(TrainingMetrics.episode.desc()).first()
                metrics = TrainingMetrics.query.order_by(TrainingMetrics.episode).all()
                scores = [m.score for m in metrics]

                if not q_table or not scores:
                    return None

                return {
                    'q_table': q_table,
                    'scores': scores,
                    'epsilon': latest_metrics.epsilon if latest_metrics else 0.1
                }
        except Exception as e:
            print(f"Error loading state from database: {e}", flush=True)
            return None