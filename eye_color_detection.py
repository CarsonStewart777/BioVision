from utils import *

def extract_eye_regions(image, landmarks):
    

    # Position of left and right eye points
    left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]

    right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
    
    # Bounding box margin

    margin = 5

    # Left eye box
   
    left_x = min(p[0] for p in left_eye_points)
    left_y = min(p[1] for p in left_eye_points)
    left_w = max(p[0] for p in left_eye_points) - left_x
    left_h = max(p[1] for p in left_eye_points) - left_y

    # Right eye box

    right_x = min(p[0] for p in right_eye_points)
    right_y = min(p[1] for p in right_eye_points)
    right_w = max(p[0] for p in right_eye_points) - right_x
    right_h = max(p[1] for p in right_eye_points) - right_y

    # Extract regions with margin

    height, width = image.shape[:2]
    left_eye_region = image[
        max(0, left_y - margin) : min(left_y + left_h + margin, height),
        max(0, left_x - margin) : min(left_x + left_w + margin, width)
    ]

    right_eye_region = image[
        max(0, right_y - margin) : min(right_y + right_h + margin, height),
        max(0, right_x - margin) : min(right_x + right_w + margin, width)
    ]

    return left_eye_region, right_eye_region

def detect_pupil(eye_region):

    if eye_region.size == 0:
        return None
    
    # Convert to grayscale
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur

    blurred = cv2.GaussianBlur(gray_eye, (5, 5), 0)

    # Apply Hough Circle Transform

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=50,
        param2=30,
        minRadius=0,
        maxRadius=min(eye_region.shape) // 2,
    )

    if circles is not None:
        # Convert to int
        circles = np.round(circles[0, :]).astype("int")

        # Return the first circle found (pupil)

        return circles[0][0], circles[0][1], circles[0][2]
    
    # if no pupil is foun, estimate the center of eye region as pupil
    return (
        eye_region.shape[1] // 2, 
        eye_region.shape[0] // 2,
    )

def create_iris_mask(eye_region, pupil_info):

    if pupil_info is none:
        return np.zeros(eye_region.shape[:2], dtype=np.uint8)
    
    center_x, center_y, pupil_radius = pupil_info

    mask = np.zeros(eye_region.shape[:2], dtype=np.uint8)

    iris_radius = int(pupil_radius * 2.5)

    cv2.circle(mask, (center_x, center_y), iris_radius, 255, -1)

    cv2.circle(mask, (center_x, center_y), pupil_radius, 0, -1)

    return mask
