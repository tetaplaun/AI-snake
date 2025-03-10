import pygame
import sys
from game.snake_game import SnakeGame
from ai.agent import QLearningAgent
from utils.visualization import MetricsVisualizer
from utils.state_manager import StateManager

def main():
    # Initialize Pygame
    pygame.init()
    pygame.font.init()

    # Create game instance
    game = SnakeGame()
    
    # Initialize AI agent
    agent = QLearningAgent(state_size=12, action_size=4)
    
    # Initialize state manager and metrics visualizer
    state_manager = StateManager()
    metrics_visualizer = MetricsVisualizer()

    # Load previous training if available
    saved_state = state_manager.load_state()
    if saved_state:
        agent.q_table = saved_state['q_table']
        metrics_visualizer.scores = saved_state['scores']
        metrics_visualizer.update_plots()

    running = True
    training = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    training = not training
                elif event.key == pygame.K_s:
                    state_manager.save_state({
                        'q_table': agent.q_table,
                        'scores': metrics_visualizer.scores
                    })

        if training:
            # Get current state
            state = game.get_state()
            
            # Get action from agent
            action = agent.get_action(state)
            
            # Execute action and get reward
            reward, done = game.step(action)
            
            # Get new state
            next_state = game.get_state()
            
            # Train agent
            agent.train(state, action, reward, next_state, done)

            if done:
                metrics_visualizer.update_score(game.score)
                game.reset()
        
        # Update display
        game.render()
        metrics_visualizer.render(game.screen)
        
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
