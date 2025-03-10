import sys
import time
import threading
from game.snake_game import SnakeGame
from ai.agent import QLearningAgent
from ai.dqn_agent import DQNAgent
from utils.visualization import MetricsVisualizer
from utils.state_manager import StateManager
from web.app import app, socketio

def run_training():
    print("Starting Snake AI Training with DQN...", flush=True)
    print("-" * 50, flush=True)

    # Create game instance
    game = SnakeGame()

    # Initialize DQN agent (replaces Q-learning agent)
    agent = DQNAgent(state_size=11, action_size=3)  # Updated state_size to match actual dimensions

    # Initialize state manager and metrics visualizer
    state_manager = StateManager()
    metrics_visualizer = MetricsVisualizer()

    # Load previous training if available
    saved_state = state_manager.load_state()
    if saved_state:
        if saved_state.get('agent_type') == 'dqn':
            agent.load('dqn_model.pth')
            metrics_visualizer.scores = saved_state['scores']
            print("Loaded previous DQN training state", flush=True)
            print(f"Continuing from episode {len(metrics_visualizer.scores)}", flush=True)
        else:
            print("Found previous Q-learning state, but starting fresh with DQN", flush=True)
    else:
        print("Starting new DQN training session", flush=True)

    episodes = 0
    save_interval = 100  # Save every 100 episodes

    try:
        while True:
            episodes += 1
            state = game.reset()
            done = False
            steps = 0

            print(f"\nEpisode {episodes} started", flush=True)

            while not done:
                try:
                    # Get action from agent
                    action = agent.get_action(state)
                    steps += 1

                    # Execute action and get reward
                    reward, done = game.step(action)

                    # Get new state
                    next_state = game.get_state()

                    # Train agent
                    agent.train(state, action, reward, next_state, done)

                    state = next_state

                    # Add small delay to make it observable
                    time.sleep(0.01)

                except Exception as e:
                    print(f"Error during episode step: {e}", flush=True)
                    break

            # Update and display metrics
            print(f"Episode {episodes} finished - Score: {game.score}, Steps: {steps}", flush=True)
            metrics_visualizer.update_score(game.score)

            # Save state periodically
            if episodes % save_interval == 0:
                state_manager.save_state({
                    'dqn_agent': agent,
                    'scores': metrics_visualizer.scores,
                    'agent_type': 'dqn'
                })
                print(f"\nSaved training state at episode {episodes}", flush=True)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final state...", flush=True)
        state_manager.save_state({
            'dqn_agent': agent,
            'scores': metrics_visualizer.scores,
            'agent_type': 'dqn'
        })
        print("Final state saved. Exiting...", flush=True)
        sys.exit(0)

if __name__ == "__main__":
    # Start training in a separate thread
    training_thread = threading.Thread(target=run_training)
    training_thread.daemon = True
    training_thread.start()

    # Start the Flask server
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)