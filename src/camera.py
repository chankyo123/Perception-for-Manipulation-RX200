#!/usr/bin/env python3

"""!
Class to represent the camera.
"""
 
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

import cv2
import time
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError
from AprilTag import *
from Perception import *
from CubeTracker import *
class Camera():
    """!
    @brief      This class describes a camera.
    """

    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720,1280)).astype(np.uint16)
        self.DepthFrameH = np.zeros((720,1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.zeros((720,1280, 3)).astype(np.uint8)


        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        self.intrinsic = Perception.get_intrinsic_matrix()
        self.invIntrinsic = np.linalg.inv(self.intrinsic)
        self.extrinsic = Perception.get_base_extrinsic_matrix()
        self.invExtrinsic = np.linalg.inv(self.extrinsic)
        self.last_click = np.array([0, 0])
        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.grid_x_points = np.arange(-450, 500, 50)
        self.grid_y_points = np.arange(-175, 525, 50)
        self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))
        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275]]
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])
        self.cubeTracker:CubeTracker = None
        self.homography=None
        self.invHomography = None
        self.intrinsic_matrix = None
        self.plane_coeffs = None
        self.z_values = None
        self.detectCube = False
        
    def set_cubeTracker(self, tracker):
        self.cubeTracker = tracker
        
    def set_detectCube(self, state=False, msg=None):
        self.detectCube = state

    def set_homography(self, H):
        self.homography = H
        self.invHomography = np.linalg.inv(H)
        
    def get_homography(self):
        return self.homography
    
    def get_invHomography(self):
        return self.invHomography
    
    def set_extrinsic(self, E):
        self.extrinsic = E
        self.invExtrinsic = np.linalg.inv(E)
        
    def get_extrinsic(self):
        return self.extrinsic
        
    def get_invExtrinsic(self):
        return self.invExtrinsic
    
    def set_intrinsic(self, K):
        self.intrinsic = K
        self.invIntrinsic = np.linalg.inv(K)
        
    def get_intrinsic(self):
        return self.intrinsic
        
    def get_invIntrinsic(self):
        return self.invIntrinsic
        
        
    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.GridFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        pass

    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        pass

    def generate_plane_depth_array(self):
        if self.plane_coeffs is None:
            return 
        y_dim, x_dim = self.DepthFrameRaw.shape[:2]

        # Create a meshgrid of x and y values
        x_values, y_values = np.meshgrid(np.arange(x_dim), np.arange(y_dim))

        # Extract the plane coefficients
        a, b, c, d = self.plane_coeffs

        # Calculate the z values for each (x, y) pair using vectorized operations
        z_values = -(a * x_values + b * y_values + d) / c
        
        self.z_values = np.flipud(z_values)
    
    
    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame and
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """
    
        if self.homography is not None:
            warped_rgb, warped_depth = Perception.transform_images(self.VideoFrame.copy(), H=self.homography, depth_img = self.DepthFrameRaw)
            self.GridFrame = warped_rgb
            self.DepthFrameH = warped_depth
            # for x, y in
            # if self.ui.radioUsr2.isChecked() and self.sm.camera.get_homography() is not None:
                
            

            #     # z = self.camera.DepthFrameH[iy][ix]
                    
            #     # inv_transformation_matrix = np.linalg.inv(self.sm.homography)
            #     # print(self.sm.homography)
            #     # # exit(0)
            #     # transformed_coords = np.array([pt.x(), pt.y(), 1])
            #     # original_coords = np.matmul(inv_transformation_matrix, transformed_coords)

            #     # ix = original_coords[0] / original_coords[2]
            #     # iy = original_coords[1] / original_coords[2]
                               
            # wx, wy, wz = Perception.uv_to_world(ix, iy, z, 
            #         invE = self.camera.get_invExtrinsic(), 
            #         invK = self.camera.get_invIntrinsic())
            
            # # if self.camera.plane_coeffs is not None:
            #     # a, b, c, d = self.camera.plane_coeffs
            #     # # Adjust wz using the plane equation
            #     # wz_plane = - (a * wx + b * wy + d) / c
            #     # print("adjusted", wz, wz_plane)
            #     # wz = wz_plane 
            # if self.camera.plane_coeffs is not None and self.ui.radioUsr2.isChecked():
            #         adjust = self.camera.z_values[int(iy)][int(ix)]
            #         wz -= adjust
            #         wz += 5
            # self.GridFrame[a] = [0, 0, 0]
            
        pass
     
    
    def is_in_center_square(self, coord, image_width=1280, image_height=720, square_size=100):
        """
        Checks if a coordinate is within a central square of specified size in the image.
        
        Args:
            coord (tuple): The (x, y) coordinate to check.
            image_width (int): The width of the image. Default is 1280.
            image_height (int): The height of the image. Default is 720.
            square_size (int): The size of the square. Default is 100.
            
        Returns:
            bool: True if the coordinate is within the center square, False otherwise.
        """
        # Calculate center of the image
        center_x, center_y = image_width // 2, image_height // 2
        
        # Calculate top-left and bottom-right coordinates of the central square
        top_left_x = center_x - square_size // 2
        top_left_y = center_y - square_size // 2
        bottom_right_x = center_x + square_size // 2
        bottom_right_y = center_y + square_size // 2
        
        # Check if coord is within these boundaries
        [x, y] = coord
        return (top_left_x <= x <= bottom_right_x) and (top_left_y <= y <= bottom_right_y)

    def drawTagsInRGBImage(self, msg:AprilTagDetectionArray, draw=True):
        """
        @brief      Draw tags from the tag detection

                    TODO: Use the tag detections output, to draw the corners/center/tagID of
                    the apriltags on the copy of the RGB image. And output the video to self.TagImageFrame.
                    Message type can be found here: /opt/ros/humble/share/apriltag_msgs/msg

                    center of the tag: (detection.centre.x, detection.centre.y) they are floats
                    id of the tag: detection.id
        """
        modified_image = self.VideoFrame.copy()       
        # Write your code here
        for det in msg.detections:
            if draw:
                AprilTag.draw_center(modified_image, det)
                AprilTag.draw_corners(modified_image, det)
                AprilTag.draw_name(modified_image, det)
        self.TagImageFrame = modified_image

class ImageListener(Node):
    def __init__(self, topic, camera):
        super().__init__('image_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image


class TagDetectionListener(Node):
    def __init__(self, topic, camera):
        super().__init__('tag_detection_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            topic,
            self.callback,
            10
        )
        self.camera = camera

    def callback(self, msg):
        self.camera.tag_detections = msg
        if np.any(self.camera.VideoFrame != 0):
            self.camera.drawTagsInRGBImage(msg)


class CameraInfoListener(Node):
    def __init__(self, topic, camera):
        super().__init__('camera_info_listener')  
        self.topic = topic
        self.tag_sub = self.create_subscription(CameraInfo, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.k, (3, 3))
        # print(self.camera.intrinsic_matrix)


class DepthListener(Node):
    def __init__(self, topic, camera):
        super().__init__('depth_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        # self.camera.DepthFrameRaw = self.camera.DepthFrameRaw / 2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_detection_topic = "/detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)
        
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(image_listener)
        self.executor.add_node(depth_listener)
        self.executor.add_node(camera_info_listener)
        self.executor.add_node(tag_detection_listener)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        try:
            while rclpy.ok():
                start_time = time.time()
                rgb_frame = self.camera.convertQtVideoFrame()
                depth_frame = self.camera.convertQtDepthFrame()
                tag_frame = self.camera.convertQtTagImageFrame()
                self.camera.projectGridInRGBImage()
                grid_frame = self.camera.convertQtGridFrame()
                if ((rgb_frame != None) & (depth_frame != None)):
                    self.updateFrame.emit(
                        rgb_frame, depth_frame, tag_frame, grid_frame)
                self.executor.spin_once() # comment this out when run this file alone.
                elapsed_time = time.time() - start_time
                sleep_time = max(0.03 - elapsed_time, 0)
                time.sleep(sleep_time)

                if __name__ == '__main__':
                    cv2.imshow(
                        "Image window",
                        cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                    cv2.imshow(
                        "Tag window",
                        cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Grid window",
                        cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(3)
                    time.sleep(0.03)
        except KeyboardInterrupt:
            pass
        
        self.executor.shutdown()
        

def main(args=None):
    rclpy.init(args=args)
    try:
        camera = Camera()
        videoThread = VideoThread(camera)
        videoThread.start()
        try:
            videoThread.executor.spin()
        finally:
            videoThread.executor.shutdown()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()