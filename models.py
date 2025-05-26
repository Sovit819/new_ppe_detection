from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime
import boto3
import os
import tempfile
import shutil
from ultralytics import YOLO

# Configuration
IS_EC2 = os.getenv('IS_EC2', 'false').lower() == 'true'
LOCAL_MODEL_PATH = "models/best.pt"

# EC2 Configuration
EC2_MODEL_PATH = "/home/ubuntu/models/best.pt"  # Path on EC2 instance

def load_model():
    """Load the YOLO model from appropriate location"""
    try:
        if IS_EC2:
            # On EC2, use the EC2 path
            model_path = EC2_MODEL_PATH
        else:
            # Local development, use local path
            model_path = LOCAL_MODEL_PATH
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Load the model
        return YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Database Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    detection_logs = relationship("DetectionLog", back_populates="user")

class DetectionLog(Base):
    __tablename__ = "detection_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    timestamp = Column(DateTime, default=datetime.now)
    image_path = Column(String)
    lab_coat_detected = Column(Boolean, default=False)
    face_mask_detected = Column(Boolean, default=False)
    gloves_detected = Column(Boolean, default=False)
    hair_cap_detected = Column(Boolean, default=False)
    shoe_detected = Column(Boolean, default=False)
    goggles_detected = Column(Boolean, default=False)
    user = relationship("User", back_populates="detection_logs")

class PPESettings(Base):
    __tablename__ = "ppe_settings"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    detect_lab_coat = Column(Boolean, default=True)
    detect_face_mask = Column(Boolean, default=True)
    detect_gloves = Column(Boolean, default=True)
    detect_hair_cap = Column(Boolean, default=True)
    detect_shoe = Column(Boolean, default=True)
    detect_goggles = Column(Boolean, default=True) 