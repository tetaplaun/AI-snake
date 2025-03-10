import logging
import time
import json
import numpy as np
from game.multiplayer_game import MultiplayerSnakeGame
from ai.agent import QLearningAgent
import sys
import os

# Add project root to path for imports (relative imports don't work well in this setup)
sys.path.append(os.path.abspath("."))
from web.app import broadcast_competition_result, socketio

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CompetitionManager:
    def __init__(self, agent1=None, agent2=None):
        """
        Initialize the competition manager with two AI agents
        If agents are not provided, new ones will be created
        """
        self.game = MultiplayerSnakeGame()
        
        # If agents are not provided, create new ones
        # For multiplayer, we need a bigger state size (19 for original + 8 for opponent detection)
        self.agent1 = agent1 if agent1 else QLearningAgent(state_size=27, action_size=3)
        self.agent2 = agent2 if agent2 else QLearningAgent(state_size=27, action_size=3)
        
        # Competition stats
        self.total_rounds = 0
        self.agent1_wins = 0
        self.agent2_wins = 0
        self.draws = 0
        self.high_scores = {
            'agent1': 0,
            'agent2': 0
        }
        
        # For tracking performance
        self.agent1_scores = []
        self.agent2_scores = []
        
    def run_competition(self, num_rounds=10, max_steps_per_round=2000):
        """
        Run a competition between the two agents for a specified number of rounds
        """
        logger.info(f"Starting competition for {num_rounds} rounds...")
        
        try:
            for round_num in range(1, num_rounds + 1):
                logger.info(f"Round {round_num}/{num_rounds} starting")
                
                try:
                    # Reset the game for a new round
                    state1, state2 = self.game.reset()
                    done = False
                    steps = 0
                    
                    # Play until one agent loses or max steps reached
                    while not done and steps < max_steps_per_round:
                        try:
                            # Get actions from both agents
                            action1 = self.agent1.get_action(state1)
                            action2 = self.agent2.get_action(state2)
                            
                            # Execute actions in the game
                            reward1, reward2, done = self.game.step(action1, action2)
                            
                            # Broadcast game state to frontend
                            self._broadcast_game_state()
                            
                            # Get new states
                            next_state1 = self.game.get_state1()
                            next_state2 = self.game.get_state2()
                            
                            # Train both agents
                            self.agent1.train(state1, action1, reward1, next_state1, done, self.game.score1)
                            self.agent2.train(state2, action2, reward2, next_state2, done, self.game.score2)
                            
                            # Update states
                            state1 = next_state1
                            state2 = next_state2
                            
                            steps += 1
                            
                            # Small delay to make visualization smoother but still responsive
                            time.sleep(0.05)  # Faster than before for more engaging gameplay
                        except Exception as e:
                            logger.error(f"Error during game step: {e}")
                            # Continue to next step if there's an error
                            break
                    
                    # Record scores
                    self.agent1_scores.append(self.game.score1)
                    self.agent2_scores.append(self.game.score2)
                    
                    # Update high scores
                    if self.game.score1 > self.high_scores['agent1']:
                        self.high_scores['agent1'] = self.game.score1
                    if self.game.score2 > self.high_scores['agent2']:
                        self.high_scores['agent2'] = self.game.score2
                    
                    # Determine winner
                    self._record_result(self.game.score1, self.game.score2)
                    
                    logger.info(f"Round {round_num} results: Agent 1: {self.game.score1}, Agent 2: {self.game.score2}")
                except Exception as e:
                    logger.error(f"Error in round {round_num}: {e}")
                    # Continue to next round if there's an error
                    continue
            
            # Record and report final results
            self._report_competition_results()
        except Exception as e:
            logger.error(f"Critical error in competition: {e}")
            # Make sure to report any results we have so far
            if self.total_rounds > 0:
                self._report_competition_results()
        
    def _record_result(self, score1, score2):
        """Record the result of a round"""
        self.total_rounds += 1
        
        if score1 > score2:
            self.agent1_wins += 1
        elif score2 > score1:
            self.agent2_wins += 1
        else:
            self.draws += 1
    
    def _report_competition_results(self):
        """Report the final results of the competition"""
        results = {
            'total_rounds': self.total_rounds,
            'agent1_wins': self.agent1_wins,
            'agent2_wins': self.agent2_wins,
            'draws': self.draws,
            'agent1_high_score': self.high_scores['agent1'],
            'agent2_high_score': self.high_scores['agent2'],
            'agent1_avg_score': sum(self.agent1_scores) / len(self.agent1_scores) if self.agent1_scores else 0,
            'agent2_avg_score': sum(self.agent2_scores) / len(self.agent2_scores) if self.agent2_scores else 0
        }
        
        logger.info("Competition Results:")
        logger.info(f"Total Rounds: {results['total_rounds']}")
        logger.info(f"Agent 1 Wins: {results['agent1_wins']} ({results['agent1_wins']/results['total_rounds']*100:.1f}%)")
        logger.info(f"Agent 2 Wins: {results['agent2_wins']} ({results['agent2_wins']/results['total_rounds']*100:.1f}%)")
        logger.info(f"Draws: {results['draws']} ({results['draws']/results['total_rounds']*100:.1f}%)")
        logger.info(f"Agent 1 High Score: {results['agent1_high_score']}")
        logger.info(f"Agent 2 High Score: {results['agent2_high_score']}")
        logger.info(f"Agent 1 Avg Score: {results['agent1_avg_score']:.2f}")
        logger.info(f"Agent 2 Avg Score: {results['agent2_avg_score']:.2f}")
        
        # Broadcast results to frontend
        broadcast_competition_result(results)
        
        return results
        
    def _broadcast_game_state(self):
        """Broadcast the current game state to the frontend via Socket.IO"""
        try:
            # Convert numpy int64 to regular Python int
            snake1_positions = [(int(x), int(y)) for x, y in self.game.snake1]
            snake2_positions = [(int(x), int(y)) for x, y in self.game.snake2]
            apple1_position = (int(self.game.apple1[0]), int(self.game.apple1[1]))
            apple2_position = (int(self.game.apple2[0]), int(self.game.apple2[1]))

            # Create game state message
            game_state = {
                'snake1': snake1_positions,
                'snake2': snake2_positions,
                'apple1': apple1_position,
                'apple2': apple2_position,
                'score1': self.game.score1,
                'score2': self.game.score2
            }
            
            # Emit the game state update
            socketio.emit('multiplayer_state_update', json.dumps(game_state))
        except Exception as e:
            logger.error(f"Error broadcasting game state: {e}")