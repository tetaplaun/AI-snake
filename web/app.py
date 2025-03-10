from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import json
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('dashboard.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected', flush=True)

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

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)