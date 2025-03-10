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
        """Broadcast current statistics to the web interface"""
        from web.app import socketio
        import json
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        from io import BytesIO
        import base64

        # Track specific failure type
        if failure_reason:
            if failure_reason == "wall":
                self.wall_collisions.append(1)
                self.self_collisions.append(0)
                self.timeouts.append(0)
            elif failure_reason == "self":
                self.wall_collisions.append(0)
                self.self_collisions.append(1)
                self.timeouts.append(0)
            elif failure_reason == "timeout":
                self.wall_collisions.append(0)
                self.self_collisions.append(0)
                self.timeouts.append(1)
        else:
            # If no failure, still append 0 to keep lengths consistent
            self.wall_collisions.append(0)
            self.self_collisions.append(0)
            self.timeouts.append(0)

        # Only create stats if we have data
        if len(self.scores) > 0:
            # Calculate moving averages for all metrics
            window_size = min(self.moving_avg_window_size, len(self.scores))
            moving_avg_scores = np.convolve(self.scores, np.ones(window_size)/window_size, mode='valid')

            # Calculate rolling failure percentages for visualization
            wall_pct_history = []
            self_pct_history = []
            timeout_pct_history = []

            # Create window-based failure percentages
            window_size = min(20, len(self.wall_collisions))
            for i in range(window_size, len(self.wall_collisions) + 1):
                recent_window = slice(i - window_size, i)
                total_failures = sum(self.wall_collisions[recent_window]) + sum(self.self_collisions[recent_window]) + sum(self.timeouts[recent_window])

                if total_failures > 0:
                    wall_pct = (sum(self.wall_collisions[recent_window]) / total_failures) * 100
                    self_pct = (sum(self.self_collisions[recent_window]) / total_failures) * 100
                    timeout_pct = (sum(self.timeouts[recent_window]) / total_failures) * 100
                else:
                    wall_pct = 0
                    self_pct = 0
                    timeout_pct = 0

                wall_pct_history.append(wall_pct)
                self_pct_history.append(self_pct)
                timeout_pct_history.append(timeout_pct)

            # Calculate failure type percentages for current totals
            total_failures = sum(self.failure_counts.values())
            failure_percentages = {
                'wall': 0,
                'self': 0,
                'timeout': 0
            }

            if total_failures > 0:
                for key, count in self.failure_counts.items():
                    failure_percentages[key] = (count / total_failures) * 100

            # Calculate self-collision rate over the last 100 episodes
            recent_self_collisions = sum(self.self_collisions[-100:]) if len(self.self_collisions) >= 100 else sum(self.self_collisions)
            recent_episodes = min(100, len(self.self_collisions))
            recent_self_collision_rate = (recent_self_collisions / recent_episodes) * 100 if recent_episodes > 0 else 0

            # Create statistics object
            stats = {
                'total_episodes': self.total_episodes,
                'max_score': self.max_score,
                'avg_score': float(np.mean(self.scores[-100:])) if len(self.scores) >= 100 else float(np.mean(self.scores)),
                'success_rate': float((self.success_count / self.total_episodes) * 100),
                'current_epsilon': float(epsilon),
                'current_learning_rate': float(learning_rate),
                'moving_avg_scores': moving_avg_scores.tolist(),
                'episode_numbers': list(range(1, len(self.scores) + 1)),
                'failure_counts': self.failure_counts,
                'failure_percentages': failure_percentages,
                'recent_self_collision_rate': recent_self_collision_rate,
                # New fields with historical data
                'rewards_history': self.rewards,
                'learning_rates_history': self.learning_rates,
                'wall_pct_history': wall_pct_history,
                'self_pct_history': self_pct_history,
                'timeout_pct_history': timeout_pct_history,
                'failure_episodes': list(range(window_size, len(self.wall_collisions) + 1))
            }

            # Emit the stats to any connected clients
            socketio.emit('training_stats', json.dumps(stats))
        else:
            # Default values for empty metrics
            default_stats = {
                'total_episodes': 0,
                'max_score': 0,
                'avg_score': 0.0,
                'success_rate': 0.0,
                'current_epsilon': float(epsilon),
                'current_learning_rate': float(learning_rate),
                'moving_avg_scores': [],
                'episode_numbers': [],
                'failure_counts': {'wall': 0, 'self': 0, 'timeout': 0},
                'failure_percentages': {'wall': 0, 'self': 0, 'timeout': 0},
                'recent_self_collision_rate': 0.0,
                'rewards_history': [],
                'learning_rates_history': [],
                'wall_pct_history': [],
                'self_pct_history': [],
                'timeout_pct_history': [],
                'failure_episodes': []
            }
            # Use the consistent event name
            socketio.emit('training_stats', json.dumps(default_stats))