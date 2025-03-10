import numpy as np
from web.app import broadcast_metrics

class MetricsVisualizer:
    def __init__(self):
        self.scores = []
        self.moving_avg_window = 20

    def update_score(self, score):
        self.scores.append(score)
        self.print_stats()
        self.broadcast_stats()

    def print_stats(self):
        if len(self.scores) > 0:
            current_score = self.scores[-1]
            avg_score = np.mean(self.scores)
            max_score = max(self.scores)

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
            print("-" * 40, flush=True)

    def broadcast_stats(self):
        if len(self.scores) > 0:
            metrics = {
                'current_score': self.scores[-1],
                'avg_score': float(np.mean(self.scores)),
                'max_score': max(self.scores),
                'episodes': len(self.scores),
                'moving_avg': float(np.mean(self.scores[-self.moving_avg_window:])) if len(self.scores) >= self.moving_avg_window else float(np.mean(self.scores))
            }
            broadcast_metrics(metrics)