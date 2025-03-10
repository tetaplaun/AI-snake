import numpy as np
from web.app import broadcast_metrics

class MetricsVisualizer:
    def __init__(self):
        self.scores = []
        self.rewards = []
        self.steps_per_apple = []
        self.successful_episodes = 0
        self.total_episodes = 0
        self.moving_avg_window = 20

    def update_score(self, score, total_reward, steps, epsilon):
        self.scores.append(score)
        self.rewards.append(total_reward)
        self.total_episodes += 1
        if score > 0:
            self.successful_episodes += 1
            self.steps_per_apple.append(steps / score)  # steps per apple collected

        self.print_stats(epsilon)
        self.broadcast_stats(epsilon)

    def print_stats(self, epsilon):
        if len(self.scores) > 0:
            current_score = self.scores[-1]
            avg_score = np.mean(self.scores)
            max_score = max(self.scores)
            success_rate = (self.successful_episodes / self.total_episodes) * 100
            avg_reward = np.mean(self.rewards[-self.moving_avg_window:]) if len(self.rewards) >= self.moving_avg_window else np.mean(self.rewards)
            avg_steps_per_apple = np.mean(self.steps_per_apple[-self.moving_avg_window:]) if len(self.steps_per_apple) >= self.moving_avg_window else (np.mean(self.steps_per_apple) if self.steps_per_apple else 0)

            # Calculate moving average
            if len(self.scores) >= self.moving_avg_window:
                moving_avg = np.mean(self.scores[-self.moving_avg_window:])
            else:
                moving_avg = avg_score

            print("\nTraining Statistics:", flush=True)
            print(f"Current Score: {current_score}", flush=True)
            print(f"Average Score: {avg_score:.2f}", flush=True)
            print(f"Moving Average (last {self.moving_avg_window}): {moving_avg:.2f}", flush=True)
            print(f"Max Score: {max_score}", flush=True)
            print(f"Episodes: {len(self.scores)}", flush=True)
            print(f"Success Rate: {success_rate:.1f}%", flush=True)
            print(f"Average Reward: {avg_reward:.2f}", flush=True)
            print(f"Exploration Rate (ε): {epsilon:.3f}", flush=True)
            print(f"Avg Steps per Apple: {avg_steps_per_apple:.1f}", flush=True)
            print("-" * 40, flush=True)

    def broadcast_stats(self, epsilon):
        if len(self.scores) > 0:
            success_rate = (self.successful_episodes / self.total_episodes) * 100
            avg_reward = np.mean(self.rewards[-self.moving_avg_window:]) if len(self.rewards) >= self.moving_avg_window else np.mean(self.rewards)
            avg_steps_per_apple = np.mean(self.steps_per_apple[-self.moving_avg_window:]) if len(self.steps_per_apple) >= self.moving_avg_window else (np.mean(self.steps_per_apple) if self.steps_per_apple else 0)

            metrics = {
                'current_score': self.scores[-1],
                'avg_score': float(np.mean(self.scores)),
                'max_score': max(self.scores),
                'episodes': len(self.scores),
                'moving_avg': float(np.mean(self.scores[-self.moving_avg_window:])) if len(self.scores) >= self.moving_avg_window else float(np.mean(self.scores)),
                'success_rate': float(success_rate),
                'avg_reward': float(avg_reward),
                'epsilon': float(epsilon),
                'avg_steps_per_apple': float(avg_steps_per_apple)
            }
            broadcast_metrics(metrics)