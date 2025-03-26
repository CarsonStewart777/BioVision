# utils.py
import cv2
import numpy as np

def load_image(image_path):
    """Load an image from path and convert to RGB"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def detect_face(image):
    """Detect a face in the image using OpenCV's Haar cascade"""
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None
    
    # Return the first face as (x, y, width, height)
    return faces[0]