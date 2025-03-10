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
        self.obstacles = self._generate_obstacles()
        self.apple = self._generate_apple()
        self.score = 0
        broadcast_game_state(self)
        return self.get_state()

    def _generate_obstacles(self):
        obstacles = set()
        center = (GRID_WIDTH // 2, GRID_HEIGHT // 2)

        while len(obstacles) < NUM_OBSTACLES:
            pos = (np.random.randint(0, GRID_WIDTH),
                  np.random.randint(0, GRID_HEIGHT))

            # Increased minimum distance from starting position from 8 to 10
            if abs(pos[0] - center[0]) + abs(pos[1] - center[1]) > 10:
                # Check if adding this obstacle would create a wall
                if not self._would_block_paths(pos, obstacles):
                    obstacles.add(pos)

        return obstacles

    def _would_block_paths(self, new_obstacle, existing_obstacles):
        # Simple check to prevent obstacles from forming walls
        # by ensuring no more than 2 obstacles are adjacent
        adjacent_count = 0
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            adj_pos = (new_obstacle[0] + dx, new_obstacle[1] + dy)
            if adj_pos in existing_obstacles:
                adjacent_count += 1
        return adjacent_count >= 2

    def _generate_apple(self):
        while True:
            apple = (np.random.randint(0, GRID_WIDTH),
                    np.random.randint(0, GRID_HEIGHT))
            if apple not in self.snake and apple not in self.obstacles:
                # Ensure the apple is not too close to obstacles
                if not any(abs(apple[0] - obs[0]) + abs(apple[1] - obs[1]) < 3 
                         for obs in self.obstacles):
                    return apple

    def get_state(self):
        head = self.snake[0]

        # Danger straight, right, left including obstacles
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

        # Nearest obstacle detection
        nearest_obstacle = self._find_nearest_obstacle(head)
        obstacle_l = nearest_obstacle[0] < head[0]
        obstacle_r = nearest_obstacle[0] > head[0]
        obstacle_u = nearest_obstacle[1] < head[1]
        obstacle_d = nearest_obstacle[1] > head[1]

        return np.array(danger + [dir_l, dir_r, dir_u, dir_d, 
                                    apple_l, apple_r, apple_u, apple_d,
                                    obstacle_l, obstacle_r, obstacle_u, obstacle_d])

    def _find_nearest_obstacle(self, pos):
        if not self.obstacles:
            return (-1, -1)  # No obstacles

        distances = [(abs(obs[0] - pos[0]) + abs(obs[1] - pos[1]), obs) 
                    for obs in self.obstacles]
        return min(distances, key=lambda x: x[0])[1]

    def _is_collision(self, pos):
        return (pos[0] < 0 or pos[0] >= GRID_WIDTH or
                pos[1] < 0 or pos[1] >= GRID_HEIGHT or
                tuple(pos) in self.snake or
                tuple(pos) in self.obstacles)

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

        # Check collision with walls, self, or obstacles
        if self._is_collision(new_head):
            return REWARD_DEATH if tuple(new_head) not in self.obstacles else REWARD_OBSTACLE, True

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
            return REWARD_MOVE, False