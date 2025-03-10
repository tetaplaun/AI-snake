import numpy as np
from web.app import broadcast_metrics

class MetricsVisualizer:
    def __init__(self):
        self.scores = []
        self.moving_avg_window = 20
        self.milestones = {
            'avg_score_5': False,    # Average score above 5
            'avg_score_10': False,   # Average score above 10
            'max_score_15': False,   # Max score above 15
            'consistent_learning': False  # Consistently improving trend
        }

    def update_score(self, score):
        self.scores.append(score)
        self.print_stats()
        self.check_milestones()
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

    def check_milestones(self):
        if len(self.scores) < self.moving_avg_window:
            return

        avg_score = np.mean(self.scores[-self.moving_avg_window:])
        max_score = max(self.scores)

        # Check for milestones
        if avg_score > 5 and not self.milestones['avg_score_5']:
            self.milestones['avg_score_5'] = True
            print("\n🎯 MILESTONE: Average score above 5 achieved!", flush=True)

        if avg_score > 10 and not self.milestones['avg_score_10']:
            self.milestones['avg_score_10'] = True
            print("\n🏆 MILESTONE: Average score above 10 achieved!", flush=True)

        if max_score > 15 and not self.milestones['max_score_15']:
            self.milestones['max_score_15'] = True
            print("\n🌟 MILESTONE: Maximum score above 15 achieved!", flush=True)

        # Check for consistent learning
        if len(self.scores) >= self.moving_avg_window * 2:
            prev_avg = np.mean(self.scores[-2*self.moving_avg_window:-self.moving_avg_window])
            current_avg = np.mean(self.scores[-self.moving_avg_window:])
            if current_avg > prev_avg * 1.2 and not self.milestones['consistent_learning']:  # 20% improvement
                self.milestones['consistent_learning'] = True
                print("\n📈 MILESTONE: Consistent learning progress detected!", flush=True)

    def broadcast_stats(self):
        if len(self.scores) > 0:
            metrics = {
                'current_score': self.scores[-1],
                'avg_score': float(np.mean(self.scores)),
                'max_score': max(self.scores),
                'episodes': len(self.scores),
                'moving_avg': float(np.mean(self.scores[-self.moving_avg_window:])) if len(self.scores) >= self.moving_avg_window else float(np.mean(self.scores)),
                'milestones': self.milestones
            }
            broadcast_metrics(metrics)