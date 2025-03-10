from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from web.app import db

class QTableEntry(db.Model):
    __tablename__ = 'q_table_entries'

    id = Column(Integer, primary_key=True)
    state_key = Column(String, nullable=False)  # String representation of state tuple
    action = Column(Integer, nullable=False)
    q_value = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    # Composite unique constraint
    __table_args__ = (db.UniqueConstraint('state_key', 'action', name='unique_state_action'),)

class TrainingMetrics(db.Model):
    __tablename__ = 'training_metrics'

    id = Column(Integer, primary_key=True)
    episode = Column(Integer, nullable=False)
    score = Column(Integer, nullable=False)
    steps = Column(Integer, nullable=False)
    total_reward = Column(Float, nullable=False)
    epsilon = Column(Float, nullable=False)
    timestamp = Column(DateTime, server_default=func.now())