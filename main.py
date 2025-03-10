import time
import threading
import logging
import numpy as np
import os
import signal
import sys
from game.snake_game import SnakeGame
from game.multiplayer_game import MultiplayerSnakeGame
from ai.agent import QLearningAgent
from ai.competition import CompetitionManager
from utils.state_manager import StateManager
from utils.visualization import MetricsVisualizer
from web.app import app, socketio

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global variables
competition_in_progress = False
training_reset_requested = False

def run_training():
    """Run the training loop in a separate thread"""
    global training_reset_requested, competition_in_progress

    try:
        logger.info("Starting Snake AI Training...")
        time.sleep(2)  # Give the server time to fully start

        # Create game instance
        game = SnakeGame()
        # Set socketio for game state broadcasting
        game.set_socketio(socketio)
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
                metrics_visualizer.scores = saved_state.get('scores', [])
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
                # Check if reset was requested at the beginning of each episode
                if training_reset_requested:
                    logger.info("RESET FLAG DETECTED - Processing training reset request...")

                    # Reset the agent to initial state
                    agent.reset()
                    logger.info("Agent reset to initial state")

                    # Reset the game
                    game.reset()
                    logger.info("Game reset")

                    # Reset metrics
                    metrics_visualizer.reset()
                    logger.info("Metrics reset")

                    # Reset episode counter
                    episodes = 0
                    logger.info("Episode counter reset to 0")

                    # Delete state file if it exists
                    try:
                        if os.path.exists("snake_ai_state.json"):
                            os.remove("snake_ai_state.json")
                            logger.info("State file deleted during reset")
                    except Exception as e:
                        logger.error(f"Error deleting state file: {e}")

                    # Broadcast empty stats to UI
                    metrics_visualizer.broadcast_stats(agent.epsilon, agent.learning_rate)
                    logger.info("Empty stats broadcast to UI")

                    # Clear the reset flag last
                    training_reset_requested = False
                    logger.info("Reset flag cleared - reset completed successfully")

                    # Skip to next iteration
                    continue

                # Skip training if competition is in progress
                if competition_in_progress:
                    time.sleep(1)
                    continue

                episodes += 1
                state = game.reset()
                done = False
                steps = 0
                total_reward = 0

                logger.info(f"\nEpisode {episodes} started")

                # Main episode loop
                while not done and not competition_in_progress:
                    # Check for reset request during episode
                    if training_reset_requested:
                        logger.info("Reset requested during episode - aborting current episode")
                        break

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

                # Skip metrics update if reset was requested
                if training_reset_requested:
                    logger.info("Reset requested - skipping metrics update")
                    continue

                # Update and display metrics
                if not competition_in_progress:
                    logger.info(f"Episode {episodes} finished - Score: {game.score}, Steps: {steps}")

                    # Get failure reason if episode ended due to failure
                    failure_reason = game.failure_reason if hasattr(game, 'failure_reason') else None

                    # Update metrics with failure reason
                    metrics_visualizer.update_score(
                        game.score,
                        total_reward,
                        steps,
                        agent.epsilon,
                        agent.learning_rate,
                        failure_reason
                    )

                    # Log specific information about the failure reason
                    if failure_reason == "self":
                        logger.info(f"Episode {episodes} ended due to SELF COLLISION - Applied stronger penalty")
                    elif failure_reason == "wall":
                        logger.info(f"Episode {episodes} ended due to wall collision")
                    elif failure_reason == "timeout":
                        logger.info(f"Episode {episodes} ended due to timeout")

                    # Save state periodically if not in reset
                    if episodes % save_interval == 0 and not training_reset_requested:
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
                time.sleep(1)
                continue

    except Exception as e:
        logger.error(f"Critical error in training loop: {e}", exc_info=True)
        return

def run_competition():
    """Run a competition between two AI agents"""
    global competition_in_progress

    try:
        logger.info("Starting Snake AI Competition...")
        competition_in_progress = True

        # Load trained agent if available
        state_manager = StateManager()
        saved_state = state_manager.load_state()

        # Create primary agent (either from saved state or new)
        agent1 = QLearningAgent(state_size=27, action_size=3)

        # Create second agent with same model
        agent2 = QLearningAgent(state_size=27, action_size=3)

        if saved_state:
            # Load the same trained state into both agents
            agent1.load_state(saved_state)
            agent2.load_state(saved_state)
            logger.info("Loaded trained model for both competing agents")
        else:
            logger.info("No saved state found, using new agents for competition")

        # Add significant variations to agent2's behavior to create more diverse gameplay
        agent2.epsilon = 0.3    # More exploration - will try more random moves

        # Modify how agent2 values different rewards to create different strategies
        # This will make it prioritize different actions than agent1
        for state in agent2.q_table:
            actions = agent2.q_table[state]
            # Randomly adjust some Q-values to create different behavior
            for i in range(len(actions)):
                # Apply a random adjustment within a certain range
                adjustment = np.random.uniform(-0.2, 0.5)
                actions[i] += adjustment

        # Create competition manager
        competition = CompetitionManager(agent1, agent2)

        # Run the competition with more rounds and steps for a better experience
        logger.info("Starting competition execution...")
        # More rounds, reasonable step limit for engaging matches
        competition.run_competition(num_rounds=15, max_steps_per_round=3000)

        # Competition complete
        logger.info("Competition completed successfully")

    except Exception as e:
        logger.error(f"Error in competition: {e}", exc_info=True)
    finally:
        # Always make sure to reset this flag
        competition_in_progress = False

@socketio.on('start_competition')
def handle_start_competition():
    global competition_in_progress
    if not competition_in_progress:
        # Notify frontend that competition is starting
        competition_in_progress = True  # Set flag before starting thread
        socketio.emit('competition_started')

        # Start competition in a separate thread
        competition_thread = threading.Thread(target=run_competition)
        competition_thread.daemon = True
        competition_thread.start()

        logger.info("Competition started and thread launched")
        return True
    logger.info("Competition already in progress, ignoring request")
    return False

@socketio.on('training_reset')
def handle_training_reset():
    global training_reset_requested
    logger.info("Training reset requested from web interface")

    # Set the flag to trigger reset in the training loop
    training_reset_requested = True

    # Try to directly remove the state file
    try:
        if os.path.exists("snake_ai_state.json"):
            os.remove("snake_ai_state.json")
            logger.info("State file deleted directly in reset handler")
    except Exception as e:
        logger.error(f"Error deleting state file in reset handler: {e}")

    logger.info("Reset flag set to True")
    return True

@socketio.on('training_reset_internal')
def handle_training_reset_internal():
    global training_reset_requested
    logger.info("Training reset requested internally")

    # Set the flag to trigger reset in the training loop
    training_reset_requested = True

    # Also try to directly remove the state file
    try:
        if os.path.exists("snake_ai_state.json"):
            os.remove("snake_ai_state.json")
            logger.info("State file deleted directly in reset handler")
    except Exception as e:
        logger.error(f"Error deleting state file in reset handler: {e}")

    logger.info("Reset flag set")
    return True

if __name__ == "__main__":
    try:
        logger.info("Starting Flask server...")

        # Start training in a background thread after a short delay
        training_thread = threading.Thread(target=run_training)
        training_thread.daemon = True
        training_thread.start()

        # Run the Flask server (this will block)
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False, log_output=True)

    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        sys.exit(1)