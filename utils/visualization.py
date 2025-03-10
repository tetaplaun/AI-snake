import pygame
import matplotlib.pyplot as plt
import numpy as np
from game.constants import *

class MetricsVisualizer:
    def __init__(self):
        self.scores = []
        self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)
        self.stats_surface = pygame.Surface((200, 100))
        self.moving_avg_window = 20

    def update_score(self, score):
        self.scores.append(score)
        self.update_plots()

    def update_plots(self):
        if len(self.scores) > 0:
            # Calculate moving average
            if len(self.scores) >= self.moving_avg_window:
                moving_avg = np.convolve(self.scores, 
                                       np.ones(self.moving_avg_window)/self.moving_avg_window, 
                                       mode='valid')
            else:
                moving_avg = np.mean(self.scores)

            # Clear previous plot
            plt.clf()
            plt.plot(self.scores, label='Score', alpha=0.5)
            if len(self.scores) >= self.moving_avg_window:
                plt.plot(range(self.moving_avg_window-1, len(self.scores)), 
                        moving_avg, label='Moving Average', color='red')
            plt.title('Training Progress')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.legend()
            plt.savefig('training_progress.png')

    def render(self, screen):
        self.stats_surface.fill(BACKGROUND_COLOR)
        
        # Render current statistics
        current_score = self.scores[-1] if self.scores else 0
        avg_score = np.mean(self.scores) if self.scores else 0
        max_score = max(self.scores) if self.scores else 0
        
        score_text = self.font.render(f'Score: {current_score}', True, TEXT_COLOR)
        avg_text = self.font.render(f'Avg: {avg_score:.1f}', True, TEXT_COLOR)
        max_text = self.font.render(f'Max: {max_score}', True, TEXT_COLOR)
        
        self.stats_surface.blit(score_text, (10, 10))
        self.stats_surface.blit(avg_text, (10, 40))
        self.stats_surface.blit(max_text, (10, 70))
        
        screen.blit(self.stats_surface, (WINDOW_WIDTH - 210, 10))
