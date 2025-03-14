from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import json
import numpy as np
import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/multiplayer')
def multiplayer():
    return render_template('multiplayer.html')

@app.route('/health')
def health_check():
    return 'OK', 200

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('reset_training')
def handle_reset():
    try:
        logger.info("Reset training requested from web interface")

        # Set the reset flag in main.py
        from main import handle_training_reset
        handle_training_reset()
        logger.info("Reset flag set in main.py")

        # Wait briefly to ensure the training loop has time to process the reset
        time.sleep(0.5)

        # Notify clients that training has been reset
        socketio.emit('training_reset')
        logger.info("UI reset signal sent")

        return True
    except Exception as e:
        logger.error(f"Error resetting training data: {e}")
        return False

@socketio.on('start_competition')
def handle_start_competition():
    logger.info("Received request to start AI competition...")
    # The actual competition starting logic is in main.py
    socketio.emit('competition_started')
    return True

def broadcast_metrics(metrics):
    socketio.emit('metrics_update', json.dumps(metrics))

def broadcast_game_state(game_state):
    # Convert numpy int64 to regular Python int
    snake_positions = [(int(x), int(y)) for x, y in game_state.snake]
    apple_position = (int(game_state.apple[0]), int(game_state.apple[1]))

    socketio.emit('game_state_update', json.dumps({
        'snake': snake_positions,
        'apple': apple_position
    }))

def broadcast_multiplayer_game_state(game_state):
    # Convert numpy int64 to regular Python int
    snake1_positions = [(int(x), int(y)) for x, y in game_state.snake1]
    snake2_positions = [(int(x), int(y)) for x, y in game_state.snake2]
    apple1_position = (int(game_state.apple1[0]), int(game_state.apple1[1]))
    apple2_position = (int(game_state.apple2[0]), int(game_state.apple2[1]))

    socketio.emit('multiplayer_state_update', json.dumps({
        'snake1': snake1_positions,
        'snake2': snake2_positions,
        'apple1': apple1_position,
        'apple2': apple2_position,
        'score1': game_state.score1,
        'score2': game_state.score2
    }))

def broadcast_competition_result(result):
    socketio.emit('competition_result', json.dumps(result))

if __name__ == '__main__':
    try:
        logger.info("Starting Flask server...")
        # ALWAYS serve the app on port 5000
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False, log_output=True)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise