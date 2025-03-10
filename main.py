import sys
import time
import threading
from game.snake_game import SnakeGame
from ai.agent import QLearningAgent
from utils.visualization import MetricsVisualizer
from utils.state_manager import StateManager
from web.app import app, socketio, db

def run_training():
    print("Starting Snake AI Training...", flush=True)
    print("-" * 50, flush=True)

    # Create game instance
    game = SnakeGame()

    # Initialize AI agent with expanded state size
    agent = QLearningAgent(state_size=19, action_size=3)  

    # Initialize state manager and metrics visualizer
    state_manager = StateManager()
    metrics_visualizer = MetricsVisualizer()

    # Load previous training if available
    saved_state = state_manager.load_state()
    if saved_state:
        agent.load_state(saved_state)
        metrics_visualizer.scores = saved_state['scores']
        print("Loaded previous training state", flush=True)
        print(f"Continuing from episode {len(metrics_visualizer.scores)}", flush=True)
        print(f"Current exploration rate (epsilon): {agent.epsilon:.3f}", flush=True)
    else:
        print("Starting new training session", flush=True)

    episodes = 0
    save_interval = 100  # Save every 100 episodes

    try:
        while True:
            episodes += 1
            state = game.reset()
            done = False
            steps = 0
            total_reward = 0

            print(f"\nEpisode {episodes} started", flush=True)

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
                agent.train(state, action, reward, next_state, done)

                state = next_state

                # Add small delay to make it observable
                time.sleep(0.01)

            # Update and display metrics
            print(f"Episode {episodes} finished - Score: {game.score}, Steps: {steps}", flush=True)
            metrics_visualizer.update_score(game.score, total_reward, steps, agent.epsilon)

            # Save state periodically
            if episodes % save_interval == 0:
                state_manager.save_state({
                    'q_table': agent.q_table,
                    'scores': metrics_visualizer.scores,
                    'steps': steps,
                    'total_reward': total_reward,
                    'epsilon': agent.epsilon
                })
                print(f"\nSaved training state at episode {episodes}", flush=True)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final state...", flush=True)
        state_manager.save_state({
            'q_table': agent.q_table,
            'scores': metrics_visualizer.scores,
            'steps': steps,
            'total_reward': total_reward,
            'epsilon': agent.epsilon
        })
        print("Final state saved. Exiting...", flush=True)
        sys.exit(0)

def save_checkpoint():
    """
    Manually save a checkpoint of the current training state
    """
    with app.app_context():
        state_manager = StateManager()
        metrics_visualizer = MetricsVisualizer()

        # Get the latest Q-table from the agent
        agent = QLearningAgent(state_size=19, action_size=3) # Updated state size to 19
        saved_state = state_manager.load_state()
        if saved_state:
            agent.load_state(saved_state)
            success = state_manager.save_state({
                'q_table': agent.q_table,
                'scores': metrics_visualizer.scores,
                'steps': 0,  # Not in active episode
                'total_reward': 0.0,  # Not in active episode
                'epsilon': agent.epsilon
            })
            if success:
                print("\nCheckpoint saved successfully!", flush=True)
                return True
    return False

if __name__ == "__main__":
    # Create database tables before starting
    with app.app_context():
        db.create_all()
        save_checkpoint()  # Save initial checkpoint

    # Start training in a separate thread
    training_thread = threading.Thread(target=run_training)
    training_thread.daemon = True
    training_thread.start()

    # Start the Flask server
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)