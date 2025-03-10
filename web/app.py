from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import json

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

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
