import cv2
import numpy as np
import dlib 

def load_image(image_path):
    ## Load an image from path and converts to RGB
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found in path: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def get_face_detector():
    ## Initializes and returns dlib face detector
    return dlib.get_frontal_face_detector()

def get_landmark_predictor():
    ## Initializes and returns dlib landmark predictor
    return dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def detect_faces(image, detector):

    return detector(image)