# Colors
SNAKE_COLOR = (46, 204, 113)  # #2ECC71
APPLE_COLOR = (231, 76, 60)   # #E74C3C
BACKGROUND_COLOR = (44, 62, 80)  # #2C3E50
GRID_COLOR = (52, 73, 94)     # #34495E
TEXT_COLOR = (236, 240, 241)  # #ECF0F1

# Game constants
GRID_SIZE = 20
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
GRID_WIDTH = WINDOW_WIDTH // GRID_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // GRID_SIZE

# AI settings
LEARNING_RATE = 0.001  # Reduced to make learning more stable
DISCOUNT_FACTOR = 0.99  # Increased to make agent more forward-thinking
EPSILON = 0.1

# Font settings
FONT_NAME = "monospace"
FONT_SIZE = 16

# Rewards
REWARD_MOVE = 0.0
REWARD_APPLE = 10.0
REWARD_DEATH = -10.0
REWARD_SELF_COLLISION = -15.0  # New penalty specifically for self collisions
REWARD_CLOSER_TO_APPLE = 0.1
REWARD_AWAY_FROM_APPLE = -0.1