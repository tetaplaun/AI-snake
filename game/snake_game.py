import pygame
import numpy as np
from collections import deque
from .constants import *

class SnakeGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Snake AI")
        self.reset()

    def reset(self):
        self.snake = deque([(GRID_WIDTH // 2, GRID_HEIGHT // 2)])
        self.direction = np.array([1, 0])
        self.apple = self._generate_apple()
        self.score = 0
        return self.get_state()

    def _generate_apple(self):
        while True:
            apple = (np.random.randint(0, GRID_WIDTH),
                    np.random.randint(0, GRID_HEIGHT))
            if apple not in self.snake:
                return apple

    def get_state(self):
        head = self.snake[0]
        
        # Danger straight, right, left
        danger = [
            self._is_collision(head + self.direction),
            self._is_collision(self._rotate_vector(self.direction, 1)),
            self._is_collision(self._rotate_vector(self.direction, -1))
        ]
        
        # Direction
        dir_l = np.array_equal(self.direction, [-1, 0])
        dir_r = np.array_equal(self.direction, [1, 0])
        dir_u = np.array_equal(self.direction, [0, -1])
        dir_d = np.array_equal(self.direction, [0, 1])
        
        # Apple location
        apple_l = self.apple[0] < head[0]
        apple_r = self.apple[0] > head[0]
        apple_u = self.apple[1] < head[1]
        apple_d = self.apple[1] > head[1]
        
        return np.array(danger + [dir_l, dir_r, dir_u, dir_d, apple_l, apple_r, apple_u, apple_d])

    def _is_collision(self, pos):
        return (pos[0] < 0 or pos[0] >= GRID_WIDTH or
                pos[1] < 0 or pos[1] >= GRID_HEIGHT or
                tuple(pos) in self.snake)

    def _rotate_vector(self, vector, rotation):
        if rotation == 1:  # right
            return np.array([vector[1], -vector[0]])
        else:  # left
            return np.array([-vector[1], vector[0]])

    def step(self, action):
        # 0: straight, 1: right, 2: left
        if action == 1:
            self.direction = self._rotate_vector(self.direction, 1)
        elif action == 2:
            self.direction = self._rotate_vector(self.direction, -1)
        
        new_head = (self.snake[0][0] + self.direction[0],
                   self.snake[0][1] + self.direction[1])
        
        # Check collision
        if self._is_collision(new_head):
            return REWARD_DEATH, True
        
        self.snake.appendleft(new_head)
        
        # Check apple
        if new_head == self.apple:
            self.score += 1
            self.apple = self._generate_apple()
            return REWARD_APPLE, False
        else:
            self.snake.pop()
            return REWARD_MOVE, False

    def render(self):
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw grid
        for x in range(0, WINDOW_WIDTH, GRID_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, WINDOW_HEIGHT))
        for y in range(0, WINDOW_HEIGHT, GRID_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (WINDOW_WIDTH, y))
        
        # Draw snake
        for segment in self.snake:
            pygame.draw.rect(self.screen, SNAKE_COLOR,
                           (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE,
                            GRID_SIZE - 1, GRID_SIZE - 1))
        
        # Draw apple
        pygame.draw.rect(self.screen, APPLE_COLOR,
                        (self.apple[0] * GRID_SIZE, self.apple[1] * GRID_SIZE,
                         GRID_SIZE - 1, GRID_SIZE - 1))
