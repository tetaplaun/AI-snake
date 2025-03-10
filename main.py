import sys
import time
import threading
import logging
from game.snake_game import SnakeGame
from ai.agent import QLearningAgent
from utils.visualization import MetricsVisualizer
from utils.state_manager import StateManager
from web.app import app, socketio, db

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def run_training():
    """Run the training loop in a separate thread"""
    try:
        logger.info("Starting Snake AI Training...")
        time.sleep(2)  # Give the server time to fully start

        # Create game instance
        game = SnakeGame()
        logger.info("Game instance created successfully")

        # Initialize AI agent with expanded state size
        agent = QLearningAgent(state_size=19, action_size=3)  
        logger.info("AI agent initialized")

        # Initialize state manager and metrics visualizer
        state_manager = StateManager()
        metrics_visualizer = MetricsVisualizer()
        logger.info("State manager and metrics visualizer initialized")

        # Load previous training if available
        try:
            saved_state = state_manager.load_state()
            if saved_state:
                agent.load_state(saved_state)
                metrics_visualizer.scores = saved_state['scores']
                logger.info("Loaded previous training state")
                logger.info(f"Continuing from episode {len(metrics_visualizer.scores)}")
                logger.info(f"Current exploration rate (epsilon): {agent.epsilon:.3f}")
            else:
                logger.info("Starting new training session")
        except Exception as e:
            logger.error(f"Error loading previous state: {e}")
            logger.info("Starting new training session due to load error")

        episodes = 0
        save_interval = 100  # Save every 100 episodes

        while True:
            try:
                episodes += 1
                state = game.reset()
                done = False
                steps = 0
                total_reward = 0

                logger.info(f"\nEpisode {episodes} started")

                while not done:
                    # Get action from agent
                    action = agent.get_action(state)
                    steps += 1

                    # Execute action and get reward
                    reward, done = game.step(action)
                    total_reward += reward

                    # Get new state
                    next_state = game.get_state()

                    # Train agent
                    agent.train(state, action, reward, next_state, done, game.score)

                    state = next_state

                    # Add small delay to make it observable
                    time.sleep(0.01)

                # Update and display metrics
                logger.info(f"Episode {episodes} finished - Score: {game.score}, Steps: {steps}")
                metrics_visualizer.update_score(game.score, total_reward, steps, agent.epsilon, agent.learning_rate)

                # Save state periodically
                if episodes % save_interval == 0:
                    try:
                        state_manager.save_state({
                            'q_table': agent.q_table,
                            'scores': metrics_visualizer.scores,
                            'steps': steps,
                            'total_reward': total_reward,
                            'epsilon': agent.epsilon,
                            'learning_rate': agent.learning_rate
                        })
                        logger.info(f"\nSaved training state at episode {episodes}")
                    except Exception as e:
                        logger.error(f"Error saving state at episode {episodes}: {e}")

            except Exception as e:
                logger.error(f"Error in episode {episodes}: {e}", exc_info=True)
                time.sleep(1)  # Brief pause before continuing to next episode
                continue

    except Exception as e:
        logger.error(f"Critical error in training loop: {e}", exc_info=True)
        return

if __name__ == "__main__":
    try:
        # Create database tables before starting
        with app.app_context():
            logger.info("Creating database tables...")
            db.create_all()
            logger.info("Database tables created successfully")

        logger.info("Starting Flask server...")

        # Start the Flask server in the main thread
        # Start training in a background thread after a short delay
        training_thread = threading.Thread(target=run_training)
        training_thread.daemon = True
        training_thread.start()

        # Run the Flask server (this will block)
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False, log_output=True)

    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        sys.exit(1)