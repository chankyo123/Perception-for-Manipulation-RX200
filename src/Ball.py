import numpy as np 
import cv2

class Ball:
    
    
    
    def __init__(self):
        pass
    
    @staticmethod
    def find(image):
        # Define the HSV range for orange
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        orange_lower = np.array([0, 183, 0])
        orange_upper = np.array([101, 239, 255])

        # Create a mask to isolate orange color
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)

        # Apply the orange mask to the image
        orange_only = cv2.bitwise_and(hsv, hsv, mask=orange_mask)

        # Convert to grayscale for circle detection
        gray = cv2.cvtColor(orange_only, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 2)

        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=30, 
            param1=50, 
            param2=30, 
            minRadius=10, 
            maxRadius=50
        )

        # Ensure some circles were found
        if circles is not None:
            # circles = np.round(circles[0, :]).astype("int")
            # for (x, y, r) in circles:
            #     # Draw the circle on the original image
            #     cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            #     cv2.circle(image, (x, y), 2, (0, 0, 255), 3)
            
            return circles
        
        return None