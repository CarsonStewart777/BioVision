# eye_color_detection.py
import cv2
import numpy as np
from utils import load_image, detect_face

def extract_eye_regions(image, face):
    """Extract eye regions using OpenCV's eye cascade"""
    # Get face coordinates
    x, y, w, h = face
    
    # Define the face ROI (Region of Interest)
    face_roi = image[y:y+h, x:x+w]
    
    # Convert to grayscale
    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
    
    # Load eye detector
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Detect eyes in the face region
    eyes = eye_cascade.detectMultiScale(gray_roi, 1.1, 3)
    
    # If we don't find exactly 2 eyes, return empty regions
    if len(eyes) != 2:
        return None, None
    
    # Sort eyes by x-coordinate (left to right)
    eyes = sorted(eyes, key=lambda eye: eye[0])
    
    # Extract each eye region with a margin
    margin = 5
    eye_regions = []
    
    for ex, ey, ew, eh in eyes:
        # Calculate eye region coordinates relative to the original image
        abs_ex = x + ex
        abs_ey = y + ey
        
        # Extract the eye region with margin
        eye_region = image[
            max(0, abs_ey - margin):min(image.shape[0], abs_ey + eh + margin),
            max(0, abs_ex - margin):min(image.shape[1], abs_ex + ew + margin)
        ]
        
        eye_regions.append(eye_region)
    
    # Return left and right eye regions
    if len(eye_regions) == 2:
        return eye_regions[0], eye_regions[1]
    else:
        return None, None

def detect_pupil(eye_region):
    """Detect the pupil in an eye region using simple circle detection"""
    if eye_region is None or eye_region.size == 0:
        return None
    
    # Convert to grayscale
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray_eye, (5, 5), 0)
    
    # Use Hough Circle Transform to find circular objects
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=eye_region.shape[0]//2,
        param1=50,
        param2=30,
        minRadius=0,
        maxRadius=eye_region.shape[0]//4
    )
    
    # If circles are found
    if circles is not None:
        # Convert to integer coordinates
        circles = np.round(circles[0, :]).astype("int")
        # Return the first circle (should be the pupil)
        return (circles[0][0], circles[0][1], circles[0][2])
    
    # If no pupil found, estimate the center of the eye region
    height, width = eye_region.shape[:2]
    return (
        width // 2,
        height // 2,
        min(height, width) // 6
    )

def create_iris_mask(eye_region, pupil_info):
    """Create a mask for the iris region"""
    if eye_region is None or pupil_info is None:
        return None
    
    center_x, center_y, pupil_radius = pupil_info
    
    # Create an empty mask the same size as the eye region
    mask = np.zeros(eye_region.shape[:2], dtype=np.uint8)
    
    # The iris is typically 2-3 times the radius of the pupil
    iris_radius = int(pupil_radius * 2.5)
    
    # Draw the iris (outer circle)
    cv2.circle(mask, (center_x, center_y), iris_radius, 255, -1)
    
    # Remove the pupil (inner circle)
    cv2.circle(mask, (center_x, center_y), pupil_radius, 0, -1)
    
    return mask

def analyze_eye_color(eye_region, iris_mask):
    """Analyze the color of the iris"""
    if eye_region is None or iris_mask is None or np.count_nonzero(iris_mask) == 0:
        return "unknown", {"blue": 0, "green": 0, "brown": 0, "hazel": 0}
    
    # Apply the mask to isolate just the iris
    iris = cv2.bitwise_and(eye_region, eye_region, mask=iris_mask)
    
    # Convert to HSV color space for better color analysis
    hsv_iris = cv2.cvtColor(iris, cv2.COLOR_RGB2HSV)
    
    # Get only the non-zero pixels (where the mask is applied)
    h_values = hsv_iris[:,:,0][iris_mask > 0]
    s_values = hsv_iris[:,:,1][iris_mask > 0]
    v_values = hsv_iris[:,:,2][iris_mask > 0]
    
    # Calculate average HSV values for the iris
    h_mean = np.mean(h_values)
    s_mean = np.mean(s_values)
    v_mean = np.mean(v_values)
    
    # Initialize confidence scores for each color
    confidence = {
        "blue": 0,
        "green": 0,
        "brown": 0,
        "hazel": 0
    }
    
    # Blue eyes: Hue range 100-140
    if 100 <= h_mean <= 140 and s_mean > 50:
        confidence["blue"] = 1.0 - abs(h_mean - 120) / 20
    
    # Green eyes: Hue range 40-80
    if 40 <= h_mean <= 80 and s_mean > 50:
        confidence["green"] = 1.0 - abs(h_mean - 60) / 30
    
    # Brown eyes: Hue range 0-20 or 170-179
    if ((0 <= h_mean <= 20) or (170 <= h_mean <= 179)) and s_mean > 50:
        confidence["brown"] = 1.0 - min(abs(h_mean - 10), abs(h_mean - 175)) / 20
    
    # Hazel eyes: Hue range 20-40
    if 20 < h_mean < 40 and s_mean > 50:
        confidence["hazel"] = 1.0 - abs(h_mean - 30) / 20
    
    # Determine the most likely color
    max_confidence = max(confidence.values())
    if max_confidence == 0:
        return "unknown", confidence
    
    for color, conf in confidence.items():
        if conf == max_confidence:
            return color, confidence

def detect_eye_color(image_path):
    """Detect eye color from an image"""
    # Load the image
    image = load_image(image_path)
    
    # Detect a face
    face = detect_face(image)
    if face is None:
        return "No face detected", {}
    
    # Extract eye regions
    left_eye, right_eye = extract_eye_regions(image, face)
    
    if left_eye is None or right_eye is None:
        return "Eyes not detected clearly", {}
    
    # Detect pupils
    left_pupil = detect_pupil(left_eye)
    right_pupil = detect_pupil(right_eye)
    
    # Create iris masks
    left_iris_mask = create_iris_mask(left_eye, left_pupil)
    right_iris_mask = create_iris_mask(right_eye, right_pupil)
    
    # Analyze eye colors
    left_color, left_confidence = analyze_eye_color(left_eye, left_iris_mask)
    right_color, right_confidence = analyze_eye_color(right_eye, right_iris_mask)
    
    # Combine results
    if left_color == right_color:
        final_color = left_color
    else:
        # If different, take the one with higher confidence
        left_max = max(left_confidence.values()) if left_confidence else 0
        right_max = max(right_confidence.values()) if right_confidence else 0
        
        if left_max > right_max:
            final_color = left_color
        else:
            final_color = right_color
    
    # Average the confidence scores
    avg_confidence = {}
    for color in left_confidence:
        avg_confidence[color] = (left_confidence.get(color, 0) + right_confidence.get(color, 0)) / 2
    
    return final_color, avg_confidence

if __name__ == "__main__":
    # Test with a sample image
    try:
        image_path = "test_images/face.jpg"  # Replace with your test image
        color, confidence = detect_eye_color(image_path)
        print(f"Detected eye color: {color}")
        print(f"Confidence scores: {confidence}")
    except Exception as e:
        print(f"Error: {e}")