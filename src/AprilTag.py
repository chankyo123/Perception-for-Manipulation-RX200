import numpy as np
from apriltag_msgs.msg import *
import cv2

class AprilTag():

    TRUE_A_LOC=[[-250, -25, 0],[250, -25, 0],[250, 275, 0],[-250, 275, 0],]
    
    FULL_A_LOC=[[-250, -25, 0],[250, -25, 0],[250, 275, 0],[-250, 275, 0],
                [25, 450, 154], [425, 50, 184], [-425, 300, 242], [-475, -150, 100]]
    TRUE_A_LOC_XY = np.array([[x, y] for x, y, z in TRUE_A_LOC])

    @staticmethod
    def get_corners(corners:AprilTagDetection.corners, tuple:bool = False):
        if tuple:
            return [(int(corner.x), int(corner.y)) for corner in corners]
        return np.array([[int(corner.x), int(corner.y)] for corner in corners])
    
    @staticmethod
    def draw_corners (image, det:AprilTagDetection):
        center = det.centre
        corners = AprilTag.get_corners(det.corners)
        # print(corners)
        # Optionally, draw lines connecting the corners to visualize the tag
        cv2.polylines(image, [corners], isClosed=True, color=(255, 0, 0), 
                      thickness=2)  # Blue lines

        return image
      
    @staticmethod  
    def draw_center (image, det:AprilTagDetection):
        center = det.centre
        cv2.circle(image, (int(center.x), int(center.y)), radius=7, 
                   color=(255, 0, 0), thickness=-1)  # Blue circle for center

    def draw_name (image, det:AprilTagDetection):
        center = det.centre
        cv2.putText(image, f'Tag ID: {str(det.id)}', 
                    (int(center.x - 20), int(center.y - 30)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, 
                    cv2.LINE_AA)
        