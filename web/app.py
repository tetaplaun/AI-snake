from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
import json
import numpy as np
import os

app = Flask(__name__)
socketio = SocketIO(app)

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_recycle': 300,
    'pool_pre_ping': True
}

# Initialize SQLAlchemy with the Flask app
db = SQLAlchemy(app)

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