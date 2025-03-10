import numpy as np
from collections import deque
from .constants import *
import sys
import logging

# Configure logging
logger = logging.getLogger(__name__)

class MultiplayerSnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        # Initialize snakes at different positions: one at 1/4, one at 3/4 of the grid
        self.snake1 = deque([((GRID_WIDTH // 4), GRID_HEIGHT // 2)])
        self.snake2 = deque([((GRID_WIDTH * 3) // 4, GRID_HEIGHT // 2)])
        
        # Snake 1 starts moving right, Snake 2 starts moving left
        self.direction1 = np.array([1, 0])
        self.direction2 = np.array([-1, 0])
        
        # Each snake has its own apple
        self.apple1 = self._generate_apple(self.snake1)
        self.apple2 = self._generate_apple(self.snake2)
        
        self.score1 = 0
        self.score2 = 0
        
        self.prev_distance_to_apple1 = self._get_distance_to_apple(self.snake1[0], self.apple1)
        self.prev_distance_to_apple2 = self._get_distance_to_apple(self.snake2[0], self.apple2)
        
        # Send initial game state to frontend - skipping broadcast for now
        # Will be handled by the competition manager
        
        # Return both snake states
        return self.get_state1(), self.get_state2()
    
    def _generate_apple(self, snake, other_snake=None):
        """Generate a new apple that's not on any snake body"""
        forbidden_positions = set(snake)
        if other_snake:
            forbidden_positions.update(other_snake)
            
        while True:
            apple = (np.random.randint(0, GRID_WIDTH),
                    np.random.randint(0, GRID_HEIGHT))
            if apple not in forbidden_positions:
                return apple

    def _get_distance_to_apple(self, head, apple):
        return abs(head[0] - apple[0]) + abs(head[1] - apple[1])
    
    def get_state1(self):
        """Get the state representation for snake 1"""
        return self._get_snake_state(self.snake1, self.direction1, self.apple1, self.snake2)
    
    def get_state2(self):
        """Get the state representation for snake 2"""
        return self._get_snake_state(self.snake2, self.direction2, self.apple2, self.snake1)

    def _get_snake_state(self, snake, direction, apple, other_snake):
        head = snake[0]

        # Danger detection (straight, right, left)
        danger_straight = self._is_collision(head + direction, snake, other_snake)
        danger_right = self._is_collision(head + self._rotate_vector(direction, 1), snake, other_snake)
        danger_left = self._is_collision(head + self._rotate_vector(direction, -1), snake, other_snake)

        # Direction one-hot encoding
        dir_l = np.array_equal(direction, [-1, 0])
        dir_r = np.array_equal(direction, [1, 0])
        dir_u = np.array_equal(direction, [0, -1])
        dir_d = np.array_equal(direction, [0, 1])

        # Apple direction relative to head
        apple_l = apple[0] < head[0]
        apple_r = apple[0] > head[0]
        apple_u = apple[1] < head[1]
        apple_d = apple[1] > head[1]

        # Detect nearby body segments (in 8 directions)
        body_positions = []
        for dx, dy in [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]:
            check_pos = (head[0] + dx, head[1] + dy)
            body_positions.append(check_pos in list(snake)[1:])
            
        # Detect nearby other snake segments (in 8 directions)
        other_snake_positions = []
        for dx, dy in [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]:
            check_pos = (head[0] + dx, head[1] + dy)
            other_snake_positions.append(check_pos in other_snake)

        # Combine all features
        state = np.array([
            danger_straight, danger_right, danger_left,  # Danger detection
            dir_l, dir_r, dir_u, dir_d,                 # Direction
            apple_l, apple_r, apple_u, apple_d,         # Apple location
            *body_positions,                            # Body segments (8 directions)
            *other_snake_positions                      # Other snake positions (8 directions)
        ])

        return state

    def _is_collision(self, pos, snake, other_snake):
        """Check if position collides with walls, own body, or other snake"""
        # Check wall collision
        if (pos[0] < 0 or pos[0] >= GRID_WIDTH or
            pos[1] < 0 or pos[1] >= GRID_HEIGHT):
            return True

        # Check self collision
        if tuple(pos) in snake:
            return True
            
        # Check collision with other snake
        if tuple(pos) in other_snake:
            return True
            
        return False

    def _rotate_vector(self, vector, rotation):
        if rotation == 1:  # right
            return np.array([vector[1], -vector[0]])
        else:  # left
            return np.array([-vector[1], vector[0]])

    def step(self, action1, action2):
        """
        Execute one time step for both snakes
        Returns rewards and done flags for both snakes
        """
        # Process snake 1's action
        reward1, done1 = self._process_snake_action(
            action1, self.snake1, self.direction1, self.apple1, 
            self.snake2, 'snake1'
        )
        
        # Process snake 2's action
        reward2, done2 = self._process_snake_action(
            action2, self.snake2, self.direction2, self.apple2, 
            self.snake1, 'snake2'
        )
        
        # Game state broadcasting is handled by the competition manager
        
        # Game is done if either snake dies or time limit reached
        done = done1 or done2
        
        return reward1, reward2, done

    def _process_snake_action(self, action, snake, direction, apple, other_snake, snake_name):
        """Process an action for a specific snake"""
        # 0: straight, 1: right, 2: left
        if action == 1:
            if snake_name == 'snake1':
                self.direction1 = self._rotate_vector(direction, 1)
            else:
                self.direction2 = self._rotate_vector(direction, 1)
        elif action == 2:
            if snake_name == 'snake1':
                self.direction1 = self._rotate_vector(direction, -1)
            else:
                self.direction2 = self._rotate_vector(direction, -1)
                
        # Update direction reference to point to the updated value
        direction = self.direction1 if snake_name == 'snake1' else self.direction2

        new_head = (snake[0][0] + direction[0], snake[0][1] + direction[1])

        # Calculate current distance to apple
        current_distance = self._get_distance_to_apple(new_head, apple)
        
        # Distance-based reward
        if snake_name == 'snake1':
            distance_reward = REWARD_CLOSER_TO_APPLE if current_distance < self.prev_distance_to_apple1 else REWARD_AWAY_FROM_APPLE
            self.prev_distance_to_apple1 = current_distance
        else:
            distance_reward = REWARD_CLOSER_TO_APPLE if current_distance < self.prev_distance_to_apple2 else REWARD_AWAY_FROM_APPLE
            self.prev_distance_to_apple2 = current_distance

        # Check collision
        if self._is_collision(new_head, snake, other_snake):
            # For competitive reasons, make death reward smaller
            return REWARD_DEATH * 0.5, True

        snake.appendleft(new_head)

        # Check apple
        if new_head == apple:
            if snake_name == 'snake1':
                self.score1 += 1
                self.apple1 = self._generate_apple(self.snake1, self.snake2)
            else:
                self.score2 += 1
                self.apple2 = self._generate_apple(self.snake2, self.snake1)
                
            return REWARD_APPLE, False
        else:
            snake.pop()
            return REWARD_MOVE + distance_reward, False