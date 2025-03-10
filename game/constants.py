# Colors
SNAKE_COLOR = (46, 204, 113)  # #2ECC71
APPLE_COLOR = (231, 76, 60)   # #E74C3C
OBSTACLE_COLOR = (142, 68, 173)  # #8E44AD
BACKGROUND_COLOR = (44, 62, 80)  # #2C3E50
GRID_COLOR = (52, 73, 94)     # #34495E
TEXT_COLOR = (236, 240, 241)  # #ECF0F1

# Game settings
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
GRID_SIZE = 20
GRID_WIDTH = WINDOW_WIDTH // GRID_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // GRID_SIZE
NUM_OBSTACLES = 5  # Reduced from 10 to 5 obstacles

# AI settings
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 0.3  # Increased from 0.1 to encourage more exploration

# Font settings
FONT_NAME = "monospace"
FONT_SIZE = 16

# Rewards
REWARD_APPLE = 25  # Increased from 15 to make food more attractive
REWARD_MOVE = -0.001  # Minimal penalty for moving
REWARD_DEATH = -10
REWARD_OBSTACLE = -10  # Same penalty as death for hitting obstacles