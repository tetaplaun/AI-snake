import numpy as np
from collections import deque
from .constants import *
from web.app import broadcast_game_state

class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = deque([(GRID_WIDTH // 2, GRID_HEIGHT // 2)])
        self.direction = np.array([1, 0])
        self.apple = self._generate_apple()
        self.score = 0
        self.prev_distance_to_apple = self._get_distance_to_apple()
        broadcast_game_state(self)
        return self.get_state()

    def _generate_apple(self):
        while True:
            apple = (np.random.randint(0, GRID_WIDTH),
                    np.random.randint(0, GRID_HEIGHT))
            if apple not in self.snake:
                return apple

    def _get_distance_to_apple(self):
        head = self.snake[0]
        return abs(head[0] - self.apple[0]) + abs(head[1] - self.apple[1])

    def get_state(self):
        head = self.snake[0]

        # Danger detection (straight, right, left)
        danger_straight = self._is_collision(head + self.direction)
        danger_right = self._is_collision(self._rotate_vector(self.direction, 1))
        danger_left = self._is_collision(self._rotate_vector(self.direction, -1))

        # Direction one-hot encoding
        dir_l = np.array_equal(self.direction, [-1, 0])
        dir_r = np.array_equal(self.direction, [1, 0])
        dir_u = np.array_equal(self.direction, [0, -1])
        dir_d = np.array_equal(self.direction, [0, 1])

        # Apple direction relative to head
        apple_l = self.apple[0] < head[0]
        apple_r = self.apple[0] > head[0]
        apple_u = self.apple[1] < head[1]
        apple_d = self.apple[1] > head[1]

        # Detect nearby body segments (in 8 directions)
        body_positions = []
        for dx, dy in [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]:
            check_pos = (head[0] + dx, head[1] + dy)
            body_positions.append(check_pos in list(self.snake)[1:])

        # Combine all features
        state = np.array([
            danger_straight, danger_right, danger_left,  # Danger detection
            dir_l, dir_r, dir_u, dir_d,                 # Direction
            apple_l, apple_r, apple_u, apple_d,         # Apple location
            *body_positions                             # Body segments (8 directions)
        ])

        return state

    def _is_collision(self, pos):
        # Check wall collision
        if (pos[0] < 0 or pos[0] >= GRID_WIDTH or
            pos[1] < 0 or pos[1] >= GRID_HEIGHT):
            return True

        # Check self collision
        if tuple(pos) in self.snake:
            # Extra penalty for colliding with body segments closer to head
            try:
                return True
            except ValueError:
                return False
        return False

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

        # Calculate current distance to apple
        current_distance = abs(new_head[0] - self.apple[0]) + abs(new_head[1] - self.apple[1])
        distance_reward = REWARD_CLOSER_TO_APPLE if current_distance < self.prev_distance_to_apple else REWARD_AWAY_FROM_APPLE
        self.prev_distance_to_apple = current_distance

        # Check collision
        if self._is_collision(new_head):
            return REWARD_DEATH, True

        self.snake.appendleft(new_head)

        # Check apple
        if new_head == self.apple:
            self.score += 1
            self.apple = self._generate_apple()
            broadcast_game_state(self)
            return REWARD_APPLE, False
        else:
            self.snake.pop()
            broadcast_game_state(self)
            return REWARD_MOVE + distance_reward, False