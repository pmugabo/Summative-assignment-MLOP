from sqlalchemy import Column, Integer, String, DateTime, func
from .database import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    prediction = Column(String)
    confidence = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String)
    accuracy = Column(Integer)
    loss = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String, default="completed")  # or "failed"