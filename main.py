from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import User, DetectionLog, PPESettings, Base
from pydantic import BaseModel
import os
import shutil
from datetime import datetime
from typing import List
import cv2
import numpy as np
from pathlib import Path
import json
from ultralytics import YOLO
import logging
import asyncio
import io

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Model configuration
LOCAL_MODEL_PATH = "models/best.pt"

# Load model
def load_model():
    try:
        if not os.path.exists(LOCAL_MODEL_PATH):
            print(f"Model file not found at {LOCAL_MODEL_PATH}")
            return None
        model = YOLO(LOCAL_MODEL_PATH)
        # Print the model's class names
        print("Model class names:", model.names)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load the model
model = load_model()
if not model:
    print("Warning: Model could not be loaded. Please check the model file and dependencies.")

class UserCreate(BaseModel):
    name: str
    email: str

class UserUpdate(BaseModel):
    name: str
    email: str

class PPEDetectionSettings(BaseModel):
    detect_lab_coat: bool = True
    detect_face_mask: bool = True
    detect_gloves: bool = True
    detect_hair_cap: bool = True
    detect_shoe: bool = True
    detect_goggles: bool = True

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving template: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading the page")

@app.get("/dashboard", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    try:
        return templates.TemplateResponse("dashboard.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving dashboard template: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading the dashboard")

@app.get("/users")
def read_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users

@app.post("/users")
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(name=user.name, email=user.email)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users/{user_id}")
def read_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.put("/users/{user_id}")
def update_user(user_id: int, user: UserUpdate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    db_user.name = user.name
    db_user.email = user.email
    db.commit()
    db.refresh(db_user)
    return db_user

@app.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    db.delete(user)
    db.commit()
    return {"message": "User deleted successfully"}

@app.post("/detect")
async def detect_ppe(
    file: UploadFile = File(...),
    settings: str = Form(...),
    db: Session = Depends(get_db)
):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded. Please ensure the model file exists at models/best.pt")
    
    try:
        # Parse settings from form data
        settings_dict = json.loads(settings)
        
        # Save uploaded file
        file_path = f"static/uploads/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Check if file is a video
        is_video = file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
        result_url = None  # Initialize result_url
        
        if is_video:
            # For videos, process multiple frames
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Could not open video file")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Process only 3 frames for faster results
            frames_to_process = 3
            frame_step = max(1, total_frames // frames_to_process)
            
            results = []
            frame_count = 0
            frame_urls = []
            processed_frames = 0
            
            while frame_count < total_frames and processed_frames < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process only selected frames
                if frame_count % frame_step == 0:
                    # Resize frame for faster processing (maintaining aspect ratio)
                    height, width = frame.shape[:2]
                    max_dimension = 640
                    if height > max_dimension or width > max_dimension:
                        scale = max_dimension / max(height, width)
                        frame = cv2.resize(frame, None, fx=scale, fy=scale)
                    
                    # Process frame with confidence threshold
                    result = model(frame, conf=0.5)  # Set confidence threshold directly
                    results.extend(result)
                    
                    # Draw bounding boxes on the frame
                    annotated_frame = frame.copy()
                    for r in result:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            class_id = int(box.cls[0])
                            class_name = r.names[class_id].lower()
                            confidence = float(box.conf[0])
                            
                            should_draw = False
                            if "mask" in class_name and settings_dict.get('detect_face_mask', True):
                                should_draw = True
                            elif "coat" in class_name and settings_dict.get('detect_lab_coat', True):
                                should_draw = True
                            elif "glove" in class_name and settings_dict.get('detect_gloves', True):
                                should_draw = True
                            elif "cap" in class_name and settings_dict.get('detect_hair_cap', True):
                                should_draw = True
                            elif "shoe" in class_name and settings_dict.get('detect_shoe', True):
                                should_draw = True
                            elif "goggle" in class_name and settings_dict.get('detect_goggles', True):
                                should_draw = True
                            
                            if should_draw:
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                label = f"{class_name}: {confidence:.2f}"
                                cv2.putText(annotated_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Save the frame with bounding boxes
                    result_frame_path = f"static/results/frame_{os.path.basename(file_path)}_{processed_frames}.jpg"
                    cv2.imwrite(result_frame_path, annotated_frame)
                    frame_urls.append(f"/static/results/frame_{os.path.basename(file_path)}_{processed_frames}.jpg")
                    processed_frames += 1
                
                frame_count += 1
            
            cap.release()
            
            # Set result_url to the first frame for video
            if frame_urls:
                result_url = frame_urls[0]
            
            # Clean up video file after processing
            try:
                os.remove(file_path)
            except:
                pass
        else:
            # For images, process the saved file with optimizations
            img = cv2.imread(file_path)
            if img is None:
                raise HTTPException(status_code=400, detail="Could not read image file")
            
            # Resize image for faster processing (maintaining aspect ratio)
            height, width = img.shape[:2]
            max_dimension = 640
            if height > max_dimension or width > max_dimension:
                scale = max_dimension / max(height, width)
                img = cv2.resize(img, None, fx=scale, fy=scale)
            
            # Process image with confidence threshold
            results = model(img, conf=0.5)  # Set confidence threshold directly
            
            # Create annotated image
            annotated_img = img.copy()
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id].lower()
                    confidence = float(box.conf[0])
                    
                    should_draw = False
                    if "mask" in class_name and settings_dict.get('detect_face_mask', True):
                        should_draw = True
                    elif "coat" in class_name and settings_dict.get('detect_lab_coat', True):
                        should_draw = True
                    elif "glove" in class_name and settings_dict.get('detect_gloves', True):
                        should_draw = True
                    elif "cap" in class_name and settings_dict.get('detect_hair_cap', True):
                        should_draw = True
                    elif "shoe" in class_name and settings_dict.get('detect_shoe', True):
                        should_draw = True
                    elif "goggle" in class_name and settings_dict.get('detect_goggles', True):
                        should_draw = True
                    
                    if should_draw:
                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(annotated_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save the result image
            result_filename = f"result_{os.path.basename(file_path)}"
            result_path = os.path.join("static", "results", result_filename)
            cv2.imwrite(result_path, annotated_img)
            result_url = f"/static/results/{result_filename}"
            
            # Clean up original file
            try:
                os.remove(file_path)
            except:
                pass
        
        # Process results
        detections = {
            "lab_coat": False,
            "face_mask": False,
            "gloves": False,
            "hair_cap": False,
            "shoe": False,
            "goggles": False
        }
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id].lower()
                confidence = float(box.conf[0])
                
                if confidence > 0.5:
                    # Only update detections if the item is enabled in settings
                    if "mask" in class_name and settings_dict.get('detect_face_mask', True):
                        detections["face_mask"] = True
                    elif "coat" in class_name and settings_dict.get('detect_lab_coat', True):
                        detections["lab_coat"] = True
                    elif "glove" in class_name and settings_dict.get('detect_gloves', True):
                        detections["gloves"] = True
                    elif "cap" in class_name and settings_dict.get('detect_hair_cap', True):
                        detections["hair_cap"] = True
                    elif "shoe" in class_name and settings_dict.get('detect_shoe', True):
                        detections["shoe"] = True
                    elif "goggle" in class_name and settings_dict.get('detect_goggles', True):
                        detections["goggles"] = True
        
        # Create detection log
        detection_log = DetectionLog(
            user_id=1,  # Default user
            image_path=file_path if not is_video else "video_processed",
            lab_coat_detected=detections["lab_coat"],
            face_mask_detected=detections["face_mask"],
            gloves_detected=detections["gloves"],
            hair_cap_detected=detections["hair_cap"],
            shoe_detected=detections["shoe"],
            goggles_detected=detections["goggles"]
        )
        
        db.add(detection_log)
        db.commit()
        db.refresh(detection_log)
        
        return {
            "message": "Detection completed",
            "log_id": detection_log.id,
            "detections": detections,
            "is_video": is_video,
            "result_image": result_url,
            "frame_urls": frame_urls if is_video else []
        }
    except Exception as e:
        # Clean up any temporary files
        try:
            if 'file_path' in locals():
                os.remove(file_path)
            if 'frame_path' in locals():
                os.remove(frame_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Error during detection: {str(e)}")

@app.get("/logs")
async def get_logs(db: Session = Depends(get_db)):
    logs = db.query(DetectionLog).order_by(DetectionLog.timestamp.desc()).all()
    return logs

@app.delete("/logs/{log_id}")
async def delete_log(log_id: int, db: Session = Depends(get_db)):
    log = db.query(DetectionLog).filter(DetectionLog.id == log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")
    
    db.delete(log)
    db.commit()
    return {"message": "Log deleted successfully"}

@app.get("/settings")
async def get_ppe_settings(db: Session = Depends(get_db)):
    settings = db.query(PPESettings).first()
    if not settings:
        settings = PPESettings()
        db.add(settings)
        db.commit()
    return {
        "detect_lab_coat": settings.detect_lab_coat,
        "detect_face_mask": settings.detect_face_mask,
        "detect_gloves": settings.detect_gloves,
        "detect_hair_cap": settings.detect_hair_cap,
        "detect_shoe": settings.detect_shoe,
        "detect_goggles": settings.detect_goggles
    }

@app.put("/settings")
async def update_ppe_settings(
    settings: PPEDetectionSettings,
    db: Session = Depends(get_db)
):
    db_settings = db.query(PPESettings).first()
    if not db_settings:
        db_settings = PPESettings()
        db.add(db_settings)
    
    db_settings.detect_lab_coat = settings.detect_lab_coat
    db_settings.detect_face_mask = settings.detect_face_mask
    db_settings.detect_gloves = settings.detect_gloves
    db_settings.detect_hair_cap = settings.detect_hair_cap
    db_settings.detect_shoe = settings.detect_shoe
    db_settings.detect_goggles = settings.detect_goggles
    
    db.commit()
    return {"message": "Settings updated successfully"}

# Add these variables at the top with other global variables
video_capture = None
is_capturing = False

class VideoCaptureRequest(BaseModel):
    ip_address: str

@app.post("/start_video")
async def start_video(request: VideoCaptureRequest):
    global video_capture, is_capturing
    
    try:
        if video_capture is not None:
            video_capture.release()

        print(f'ip_address: {request.ip_address}')
        
        video_capture = cv2.VideoCapture(request.ip_address)
        if not video_capture.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video stream")
        
        is_capturing = True
        return {"message": "Video capture started successfully"}
    except Exception as e:
        if video_capture is not None:
            video_capture.release()
            video_capture = None
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_frame")
async def get_frame():
    global video_capture, is_capturing
    
    if not is_capturing or video_capture is None:
        logger.error("Video capture not started or video_capture is None")
        raise HTTPException(status_code=400, detail="Video capture not started")
    
    try:
        ret, frame = video_capture.read()
        if not ret:
            logger.error("Could not read frame from video capture")
            raise HTTPException(status_code=400, detail="Could not read frame")
        
        # Log frame dimensions
        logger.debug(f"Frame dimensions: {frame.shape}")
        
        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/jpeg"
        )
    except Exception as e:
        logger.error(f"Error in get_frame: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop_video")
async def stop_video():
    global video_capture, is_capturing
    
    try:
        if video_capture is not None:
            video_capture.release()
            video_capture = None
        is_capturing = False
        return {"message": "Video capture stopped successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 