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
        # Make sure grid constants are reasonable (snakes can't be outside the visible area)
        grid_width = min(GRID_WIDTH, 40)  # Max 40 for visibility
        grid_height = min(GRID_HEIGHT, 30)  # Max 30 for visibility
        
        # Initialize snakes at different positions with starting length of 3
        # Position snakes with enough space from walls
        start_x1 = max(3, grid_width // 4)
        start_x2 = min(grid_width - 4, grid_width * 3 // 4)
        
        self.snake1 = deque([
            (start_x1, grid_height // 2),
            (start_x1 - 1, grid_height // 2),
            (start_x1 - 2, grid_height // 2)
        ])
        
        self.snake2 = deque([
            (start_x2, grid_height // 2),
            (start_x2 + 1, grid_height // 2),
            (start_x2 + 2, grid_height // 2)
        ])
        
        # Snake 1 starts moving right, Snake 2 starts moving left
        self.direction1 = np.array([1, 0])
        self.direction2 = np.array([-1, 0])
        
        # Generate fixed apples in clearly visible positions away from snakes
        # Make sure they're not too close to walls for better visibility
        self.apple1 = (max(2, grid_width // 5), max(2, grid_height // 5))
        self.apple2 = (min(grid_width - 3, grid_width * 4 // 5), 
                      min(grid_height - 3, grid_height * 4 // 5))
        
        self.score1 = 0
        self.score2 = 0
        
        self.prev_distance_to_apple1 = self._get_distance_to_apple(self.snake1[0], self.apple1)
        self.prev_distance_to_apple2 = self._get_distance_to_apple(self.snake2[0], self.apple2)
        
        logger.info(f"Game reset - Snake1 at {self.snake1[0]}, Snake2 at {self.snake2[0]}")
        logger.info(f"Apple1 at {self.apple1}, Apple2 at {self.apple2}")
        
        # Return both snake states
        return self.get_state1(), self.get_state2()
    
    def _generate_apple(self, snake, other_snake=None):
        """Generate a new apple that's not on any snake body"""
        forbidden_positions = set(tuple(pos) for pos in snake)
        if other_snake:
            forbidden_positions.update(tuple(pos) for pos in other_snake)
            
        # Generate new apple position
        for _ in range(100):  # Try 100 times to avoid infinite loop
            apple = (np.random.randint(0, GRID_WIDTH),
                    np.random.randint(0, GRID_HEIGHT))
            if apple not in forbidden_positions:
                return apple
                
        # Fallback if we couldn't find an empty spot (very rare)
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                pos = (x, y)
                if pos not in forbidden_positions:
                    return pos
                    
        # Extreme fallback (only if grid is completely full)
        return (GRID_WIDTH // 2, GRID_HEIGHT // 2)

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
                # Generate new apple in a fixed position that's visible
                grid_width = min(GRID_WIDTH, 40)
                grid_height = min(GRID_HEIGHT, 30)
                
                if self.score1 % 2 == 0:  # Even score
                    self.apple1 = (max(2, grid_width // 5), min(grid_height - 3, grid_height * 4 // 5))
                else:  # Odd score
                    self.apple1 = (min(grid_width - 3, grid_width * 2 // 5), max(2, grid_height // 5))
                
                logger.info(f"Snake1 got apple. New apple at {self.apple1}")
            else:
                self.score2 += 1
                # Generate new apple in a fixed position that's visible
                grid_width = min(GRID_WIDTH, 40)
                grid_height = min(GRID_HEIGHT, 30)
                
                if self.score2 % 2 == 0:  # Even score
                    self.apple2 = (min(grid_width - 3, grid_width * 4 // 5), max(2, grid_height // 5))
                else:  # Odd score
                    self.apple2 = (min(grid_width - 3, grid_width * 3 // 5), min(grid_height - 3, grid_height * 4 // 5))
                    
                logger.info(f"Snake2 got apple. New apple at {self.apple2}")
                
            return REWARD_APPLE, False
        else:
            snake.pop()
            return REWARD_MOVE + distance_reward, False