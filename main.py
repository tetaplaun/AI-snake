import sys
import time
from game.snake_game import SnakeGame
from ai.agent import QLearningAgent
from utils.visualization import MetricsVisualizer
from utils.state_manager import StateManager

def main():
    print("Starting Snake AI Training...", flush=True)
    print("-" * 50, flush=True)

    # Create game instance
    game = SnakeGame()

    # Initialize AI agent
    agent = QLearningAgent(state_size=12, action_size=3)

    # Initialize state manager and metrics visualizer
    state_manager = StateManager()
    metrics_visualizer = MetricsVisualizer()

    # Load previous training if available
    saved_state = state_manager.load_state()
    if saved_state:
        agent.q_table = saved_state['q_table']
        metrics_visualizer.scores = saved_state['scores']
        print("Loaded previous training state", flush=True)
        print(f"Continuing from episode {len(metrics_visualizer.scores)}", flush=True)
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

            print(f"\nEpisode {episodes} started", flush=True)

            while not done:
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

            # Update and display metrics
            print(f"Episode {episodes} finished - Score: {game.score}, Steps: {steps}", flush=True)
            metrics_visualizer.update_score(game.score)

            # Save state periodically
            if episodes % save_interval == 0:
                state_manager.save_state({
                    'q_table': agent.q_table,
                    'scores': metrics_visualizer.scores
                })
                print(f"\nSaved training state at episode {episodes}", flush=True)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final state...", flush=True)
        state_manager.save_state({
            'q_table': agent.q_table,
            'scores': metrics_visualizer.scores
        })
        print("Final state saved. Exiting...", flush=True)
        sys.exit(0)

if __name__ == "__main__":
    main()