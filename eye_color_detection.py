# eye_color_detection.py
import cv2
import numpy as np
from utils import load_image, detect_face

def detect_pupil(eye_region):
    """Detect the pupil using a simpler and more reliable method"""
    if eye_region is None or eye_region.size == 0:
        return None
    
    # Get dimensions
    height, width = eye_region.shape[:2]
    
    # The pupil is typically in the center of the eye horizontally
    # and slightly above the vertical center
    
    # Estimate pupil position based on eye anatomy
    pupil_x = width // 2
    pupil_y = int(height * 0.45)  # Slightly above center
    
    # Estimate pupil radius (typically about 1/6 to 1/8 of eye width)
    pupil_radius = width // 7
    
    # Create a region of interest around estimated pupil position
    roi_size = pupil_radius * 2
    roi_left = max(0, pupil_x - roi_size)
    roi_top = max(0, pupil_y - roi_size)
    roi_right = min(width, pupil_x + roi_size)
    roi_bottom = min(height, pupil_y + roi_size)
    
    roi = eye_region[roi_top:roi_bottom, roi_left:roi_right]
    
    # Convert to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    
    # Find the darkest point in the ROI
    min_val, _, min_loc, _ = cv2.minMaxLoc(gray_roi)
    
    # Adjust coordinates relative to the original eye region
    pupil_x = roi_left + min_loc[0]
    pupil_y = roi_top + min_loc[1]
    
    return (pupil_x, pupil_y, pupil_radius)

def extract_eye_regions(image, face):
    """Extract eye regions ensuring better eye positioning"""
    # Get face coordinates
    x, y, w, h = face
    
    # Define approximate eye regions based on facial proportions
    eye_width = w // 4
    eye_height = h // 8
    
    # Vertical position (approximately 40-45% down from the top of the face)
    eye_y = y + int(h * 0.42)
    
    # Left eye position (approximately 30% across from left edge)
    left_eye_x = x + int(w * 0.25)
    
    # Right eye position (approximately 70% across from left edge)
    right_eye_x = x + int(w * 0.70)
    
    # Calculate final coordinates with margins
    margin = 10
    left_eye = image[
        max(0, eye_y - eye_height//2 - margin):min(image.shape[0], eye_y + eye_height//2 + margin),
        max(0, left_eye_x - eye_width//2 - margin):min(image.shape[1], left_eye_x + eye_width//2 + margin)
    ]
    
    right_eye = image[
        max(0, eye_y - eye_height//2 - margin):min(image.shape[0], eye_y + eye_height//2 + margin),
        max(0, right_eye_x - eye_width//2 - margin):min(image.shape[1], right_eye_x + eye_width//2 + margin)
    ]
    
    return left_eye, right_eye



def create_iris_mask(eye_region, pupil_info):
    """Create a mask for the iris region with better proportions"""
    if eye_region is None or pupil_info is None:
        return None
    
    center_x, center_y, pupil_radius = pupil_info
    
    # Create an empty mask the same size as the eye region
    mask = np.zeros(eye_region.shape[:2], dtype=np.uint8)
    
    # The iris is typically about 3.5-4 times the radius of the pupil
    # This ratio works better with real human eye proportions
    iris_radius = int(pupil_radius * 3.8)
    
    # Limit the iris radius to avoid going too far outside the eye
    max_radius = min(eye_region.shape[0], eye_region.shape[1]) // 2
    iris_radius = min(iris_radius, max_radius)
    
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


def visualize_detection(image_path):
    """Visualize the eye detection steps"""
    # Import matplotlib here to avoid import errors
    import matplotlib.pyplot as plt
    
    # Load image
    image = load_image(image_path)
    original = image.copy()
    
    # Detect face
    face = detect_face(image)
    if face is None:
        print("No face detected")
        plt.figure(figsize=(8, 6))
        plt.imshow(original)
        plt.title("Original - No Face Detected")
        plt.axis('off')
        plt.show()
        return
    
    # Draw face rectangle
    x, y, w, h = face
    face_img = original.copy()
    cv2.rectangle(face_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Show original and face detection
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(face_img)
    plt.title("Face Detected")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Extract eyes
    left_eye, right_eye = extract_eye_regions(image, face)
    
    if left_eye is None and right_eye is None:
        print("Eyes not detected clearly")
        return
    
    # Show extracted eye regions
    plt.figure(figsize=(12, 6))
    
    if left_eye is not None and left_eye.size > 0:
        plt.subplot(1, 2, 1)
        plt.imshow(left_eye)
        plt.title("Left Eye Region")
        plt.axis('off')
    else:
        print("Left eye not detected")
    
    if right_eye is not None and right_eye.size > 0:
        plt.subplot(1, 2, 2)
        plt.imshow(right_eye)
        plt.title("Right Eye Region")
        plt.axis('off')
    else:
        print("Right eye not detected")
    
    plt.tight_layout()
    plt.show()
    
    # Process each eye
    for i, eye in enumerate([left_eye, right_eye]):
        eye_name = "Left" if i == 0 else "Right"
        if eye is None or eye.size == 0:
            continue
            
        # Detect pupil
        pupil = detect_pupil(eye)
        if pupil is None:
            print(f"No pupil detected in {eye_name.lower()} eye")
            continue
            
        # Create iris mask
        iris_mask = create_iris_mask(eye, pupil)
        
        # Show pupil and iris detection
        plt.figure(figsize=(15, 5))
        
        # Original eye
        plt.subplot(1, 3, 1)
        plt.imshow(eye)
        plt.title(f"{eye_name} Eye")
        plt.axis('off')
        
        # Eye with pupil circle drawn
        eye_with_pupil = eye.copy()
        cv2.circle(eye_with_pupil, (pupil[0], pupil[1]), pupil[2], (255, 0, 0), 2)
        plt.subplot(1, 3, 2)
        plt.imshow(eye_with_pupil)
        plt.title(f"Pupil Detection")
        plt.axis('off')
        
        # Show iris mask
        plt.subplot(1, 3, 3)
        plt.imshow(iris_mask, cmap='gray')
        plt.title(f"Iris Mask")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Analyze eye color
        color, confidence = analyze_eye_color(eye, iris_mask)
        print(f"{eye_name} eye color: {color}")
        print(f"Confidence scores: {confidence}")

if __name__ == "__main__":
    # Test with a sample image
    try:
        image_path = "test_images/face.jpg"  # Replace with your test image
        print("Running eye color detection with visualization...")
        visualize_detection(image_path)
        
        # Also run the regular detection to get the final result
        color, confidence = detect_eye_color(image_path)
        print("\nFinal result:")
        print(f"Detected eye color: {color}")
        print(f"Confidence scores: {confidence}")
    except Exception as e:
        print(f"Error: {e}")