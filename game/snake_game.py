import pygame
import numpy as np
import random
from collections import deque
from .constants import *

class SnakeGame:
    def __init__(self):
        # Use constants from constants.py
        self.grid_size = GRID_SIZE
        self.window_width = WINDOW_WIDTH
        self.window_height = WINDOW_HEIGHT
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT

        # Track failure reason
        self.failure_reason = None

        # For visualization
        self.socketio = None

        # Track collision flash effect
        self.collision_flash = None  # Will store [x, y, opacity] when collision occurs

        # Initialize game state
        self.reset()

    def set_socketio(self, socketio):
        """Set the socketio instance for broadcasting game state"""
        self.socketio = socketio

    def broadcast_state(self):
        """Broadcast the current game state to the web interface"""
        if self.socketio:
            import json
            # Ensure collision_flash is JSON serializable
            collision_flash_json = None
            if self.collision_flash:
                collision_flash_json = [float(self.collision_flash[0]),
                                        float(self.collision_flash[1]),
                                        float(self.collision_flash[2])]

            game_state = {
                'snake': self.snake.tolist(),
                'apple': self.apple.tolist(),
                'score': self.score,
                'collision_flash': collision_flash_json
            }
            self.socketio.emit('game_state_update', json.dumps(game_state))

            # Fade out the collision flash if it exists
            if self.collision_flash:
                self.collision_flash[2] -= 0.1  # Reduce opacity
                if self.collision_flash[2] <= 0:
                    self.collision_flash = None  # Remove flash when fully faded

    def reset(self):
        # Initialize snake in the middle of the grid
        self.snake = np.array([[self.grid_width // 2, self.grid_height // 2]])
        self.direction = np.array([1, 0])  # Start moving right
        self.apple = self._generate_apple()
        self.score = 0
        self.steps_without_eating = 0
        self.max_steps_without_eating = 100  # Timeout limit
        self.done = False
        self.failure_reason = None
        self.prev_distance_to_apple = self._get_distance_to_apple()
        self.collision_flash = None  # Reset collision flash

        # Broadcast initial state
        self.broadcast_state()

        return self.get_state()

    def _generate_apple(self):
        """Generate a new apple position"""
        while True:
            apple = np.array([
                np.random.randint(0, self.grid_width),
                np.random.randint(0, self.grid_height)
            ])
            # Check if apple is not on snake
            if not any(np.array_equal(apple, segment) for segment in self.snake):
                return apple

    def _get_distance_to_apple(self):
        """Calculate Manhattan distance to apple"""
        head = self.snake[0]
        return abs(head[0] - self.apple[0]) + abs(head[1] - self.apple[1])

    def get_state(self):
        """Get the current state representation for the agent"""
        head = self.snake[0]

        # Danger straight, right, left
        danger_straight = self._is_danger(self.direction)
        danger_right = self._is_danger(self._turn_right(self.direction))
        danger_left = self._is_danger(self._turn_left(self.direction))

        # Direction
        dir_up = np.array_equal(self.direction, [0, -1])
        dir_right = np.array_equal(self.direction, [1, 0])
        dir_down = np.array_equal(self.direction, [0, 1])
        dir_left = np.array_equal(self.direction, [-1, 0])

        # Apple location
        apple_up = self.apple[1] < head[1]
        apple_right = self.apple[0] > head[0]
        apple_down = self.apple[1] > head[1]
        apple_left = self.apple[0] < head[0]

        # Body positions (8 surrounding cells)
        body_positions = []
        for dx, dy in [(0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1)]:
            check_pos = np.array([head[0] + dx, head[1] + dy])
            # Check if position is part of snake body (not head)
            is_body = any(np.array_equal(check_pos, segment) for segment in self.snake[1:])
            body_positions.append(is_body)

        # Combine all features
        state = np.array([
            danger_straight, danger_right, danger_left,
            dir_up, dir_right, dir_down, dir_left,
            apple_up, apple_right, apple_down, apple_left,
            *body_positions
        ], dtype=np.int32)

        return state

    def _is_danger(self, direction):
        """Check if moving in a direction would result in danger"""
        head = self.snake[0]
        new_pos = head + direction

        # Check wall collision
        if (new_pos[0] < 0 or new_pos[0] >= self.grid_width or
            new_pos[1] < 0 or new_pos[1] >= self.grid_height):
            return True

        # Check self collision
        return any(np.array_equal(new_pos, segment) for segment in self.snake[1:])

    def _turn_right(self, direction):
        """Turn direction 90 degrees right"""
        return np.array([direction[1], -direction[0]])

    def _turn_left(self, direction):
        """Turn direction 90 degrees left"""
        return np.array([-direction[1], direction[0]])

    def _is_collision(self, pos):
        """Check if position results in collision"""
        # Check wall collision
        if (pos[0] < 0 or pos[0] >= self.grid_width or
            pos[1] < 0 or pos[1] >= self.grid_height):
            self.failure_reason = "wall"
            # Record collision point for flash effect
            # Clamp position to grid boundaries
            flash_pos = np.array(pos, dtype=np.float64)  # Ensure it's a numpy array
            if flash_pos[0] < 0:
                flash_pos[0] = 0
            elif flash_pos[0] >= self.grid_width:
                flash_pos[0] = self.grid_width - 1
            if flash_pos[1] < 0:
                flash_pos[1] = 0
            elif flash_pos[1] >= self.grid_height:
                flash_pos[1] = self.grid_height - 1
            self.collision_flash = [float(flash_pos[0]), float(flash_pos[1]), 1.0]  # [x, y, opacity]
            return True

        # Check self collision
        if any(np.array_equal(pos, segment) for segment in self.snake[1:]):
            self.failure_reason = "self"
            return True

        return False

    def step(self, action):
        """Take a step in the environment"""
        # 0: straight, 1: right, 2: left
        if action == 1:
            self.direction = self._turn_right(self.direction)
        elif action == 2:
            self.direction = self._turn_left(self.direction)

        # Move snake
        head = self.snake[0]
        new_head = head + self.direction

        # Increment step counter
        self.steps_without_eating += 1

        # Check for timeout
        if self.steps_without_eating >= self.max_steps_without_eating:
            self.failure_reason = "timeout"
            return REWARD_DEATH, True

        # Calculate current distance to apple
        current_distance = abs(new_head[0] - self.apple[0]) + abs(new_head[1] - self.apple[1])
        distance_reward = REWARD_CLOSER_TO_APPLE if current_distance < self.prev_distance_to_apple else REWARD_AWAY_FROM_APPLE
        self.prev_distance_to_apple = current_distance

        # Check collision
        if self._is_collision(new_head):
            return REWARD_DEATH, True

        # Add new head
        self.snake = np.vstack([new_head, self.snake])

        # Check apple
        if np.array_equal(new_head, self.apple):
            self.score += 1
            self.steps_without_eating = 0  # Reset steps counter
            self.apple = self._generate_apple()
            self.broadcast_state()
            return REWARD_APPLE, False
        else:
            # Remove tail
            self.snake = self.snake[:-1]
            self.broadcast_state()
            return REWARD_MOVE + distance_reward, False