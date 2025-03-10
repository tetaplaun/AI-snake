import numpy as np

class MetricsVisualizer:
    def __init__(self):
        """Initialize metrics visualizer"""
        self.scores = []
        self.rewards = []
        self.steps = []
        self.epsilons = []
        self.learning_rates = []
        self.wall_collisions = []
        self.self_collisions = []
        self.timeouts = []

        self.max_score = 0
        self.total_episodes = 0
        self.success_count = 0
        self.moving_avg_window_size = 20  # This is an integer

        # Track failure reasons
        self.failure_counts = {
            'wall': 0,
            'self': 0,
            'timeout': 0
        }

        print("Metrics visualizer initialized", flush=True)

    def reset(self):
        """Reset all metrics to initial state"""
        print("Resetting all metrics to initial state", flush=True)

        # Clear all arrays
        self.scores = []
        self.rewards = []
        self.steps = []
        self.epsilons = []
        self.learning_rates = []
        self.wall_collisions = []
        self.self_collisions = []
        self.timeouts = []

        # Reset statistics
        self.max_score = 0
        self.total_episodes = 0
        self.success_count = 0
        # moving_avg_window_size remains the same

        # Reset failure counters
        self.failure_counts = {
            'wall': 0,
            'self': 0,
            'timeout': 0
        }

        # Broadcast empty metrics to UI
        self.broadcast_stats(0.1, 0.001)  # Default values

        print("All metrics have been reset", flush=True)
        return True

    def update_score(self, score, total_reward, steps, epsilon, learning_rate, failure_reason=None):
        """Update metrics with a new score"""
        self.scores.append(score)
        self.rewards.append(total_reward)
        self.steps.append(steps)
        self.epsilons.append(epsilon)
        self.learning_rates.append(learning_rate)

        # Update max score if needed
        if score > self.max_score:
            self.max_score = score

        # Increment episode counter
        self.total_episodes += 1

        # Track failure reason if provided
        if failure_reason and failure_reason in self.failure_counts:
            self.failure_counts[failure_reason] += 1

        # Track successful episodes (where score > 0)
        if score > 0:
            self.success_count += 1

        # Print and broadcast stats
        self.print_stats(epsilon, learning_rate)
        self.broadcast_stats(epsilon, learning_rate, failure_reason)

    def print_stats(self, epsilon, learning_rate):
        """Print current training statistics"""
        if len(self.scores) > 0:
            avg_score = np.mean(self.scores)
            max_score = max(self.scores)
            success_rate = (self.success_count / self.total_episodes) * 100
            avg_reward = np.mean(self.rewards[-self.moving_avg_window_size:]) if len(self.rewards) >= self.moving_avg_window_size else np.mean(self.rewards)
            avg_steps = np.mean(self.steps[-self.moving_avg_window_size:]) if len(self.steps) >= self.moving_avg_window_size else np.mean(self.steps)
            avg_epsilon = np.mean(self.epsilons[-self.moving_avg_window_size:]) if len(self.epsilons) >= self.moving_avg_window_size else np.mean(self.epsilons)
            avg_learning_rate = np.mean(self.learning_rates[-self.moving_avg_window_size:]) if len(self.learning_rates) >= self.moving_avg_window_size else np.mean(self.learning_rates)

            # Calculate moving average
            moving_avg = np.mean(self.scores[-self.moving_avg_window_size:]) if len(self.scores) >= self.moving_avg_window_size else np.mean(self.scores)

            print("\n" + "=" * 40, flush=True)
            print(f"Episode: {self.total_episodes}", flush=True)
            print(f"Current Score: {self.scores[-1]}", flush=True)
            print(f"Average Score: {avg_score:.2f}", flush=True)
            print(f"Max Score: {max_score}", flush=True)
            print(f"Moving Avg ({self.moving_avg_window_size}): {moving_avg:.2f}", flush=True)
            print(f"Success Rate: {success_rate:.1f}%", flush=True)
            print(f"Average Reward: {avg_reward:.2f}", flush=True)
            print(f"Exploration Rate (ε): {avg_epsilon:.3f}", flush=True)
            print(f"Learning Rate (α): {avg_learning_rate:.6f}", flush=True)
            print(f"Avg Steps: {avg_steps:.1f}", flush=True)
            print("-" * 40, flush=True)

    def broadcast_stats(self, epsilon, learning_rate, failure_reason=None):
        """Broadcast metrics to the web interface"""
        # Import here to avoid circular imports
        from web.app import socketio
        import json

        if len(self.scores) > 0:
            success_rate = (self.success_count / self.total_episodes) * 100
            avg_reward = np.mean(self.rewards[-self.moving_avg_window_size:]) if len(self.rewards) >= self.moving_avg_window_size else np.mean(self.rewards)
            avg_steps = np.mean(self.steps[-self.moving_avg_window_size:]) if len(self.steps) >= self.moving_avg_window_size else np.mean(self.steps)
            avg_epsilon = np.mean(self.epsilons[-self.moving_avg_window_size:]) if len(self.epsilons) >= self.moving_avg_window_size else np.mean(self.epsilons)
            avg_learning_rate = np.mean(self.learning_rates[-self.moving_avg_window_size:]) if len(self.learning_rates) >= self.moving_avg_window_size else np.mean(self.learning_rates)
            moving_avg = np.mean(self.scores[-self.moving_avg_window_size:]) if len(self.scores) >= self.moving_avg_window_size else np.mean(self.scores)

            metrics = {
                'current_score': int(self.scores[-1]),
                'avg_score': float(np.mean(self.scores)),
                'max_score': int(self.max_score),
                'episodes': int(self.total_episodes),
                'moving_avg': float(moving_avg),
                'success_rate': float(success_rate),
                'avg_reward': float(avg_reward),
                'epsilon': float(avg_epsilon),
                'learning_rate': float(avg_learning_rate),
                'avg_steps': float(avg_steps),
                'avg_steps_per_apple': float(avg_steps / max(1, self.scores[-1])) if self.scores[-1] > 0 else 0.0,
                'failure_reason': failure_reason
            }

            # Emit directly using socketio
            socketio.emit('metrics_update', json.dumps(metrics))
        else:
            # Default values for empty metrics
            metrics = {
                'current_score': 0,
                'avg_score': 0.0,
                'max_score': 0,
                'episodes': 0,
                'moving_avg': 0.0,
                'success_rate': 0.0,
                'avg_reward': 0.0,
                'epsilon': float(epsilon),
                'learning_rate': float(learning_rate),
                'avg_steps': 0.0,
                'avg_steps_per_apple': 0.0,
                'failure_reason': None
            }

            # Emit directly using socketio
            socketio.emit('metrics_update', json.dumps(metrics))