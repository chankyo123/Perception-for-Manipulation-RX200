import numpy as np
from CameraParameters import *
from DaWorld import *
from AprilTag import *
import time, cv2
class Perception:
    @staticmethod
    def get_base_extrinsic_matrix():
        E = np.hstack((DaWorld.get_cam_to_world_rotation(),
                       DaWorld.get_cam_to_world_translation()))  # Combine R and t
        # print(E.shape)
        # time.sleep(40)
        E = np.vstack((E, [0, 0, 0, 1])) 
        return E
       
    @staticmethod 
    def get_base_invExtrinsic_matrix():
        return np.linalg.inv(Perception.get_base_extrinsic_matrix())
    
    @staticmethod
    def get_intrinsic_matrix():
        return CameraParameters.get_intrinsic_matrix()
    
    @staticmethod
    def get_invIntrinsic_matrix():
        return np.linalg.inv(Perception.get_intrinsic_matrix())
    
    @staticmethod
    def world_to_uv(world_coords, E=get_base_extrinsic_matrix(), K=get_intrinsic_matrix()):
        
        # Convert world coordinates to homogeneous form
        world_homogeneous = np.array([[world_coords[0]], [world_coords[1]], [world_coords[2]], [1]])

        # Convert world coordinates to camera coordinates
        camera_coords = E @ world_homogeneous
        
        # Extract camera coordinates (Xc, Yc, Zc)
        X_c, Y_c, Z_c = camera_coords[0, 0], camera_coords[1, 0], camera_coords[2, 0]

        # Project camera coordinates to image coordinates
        image_homogeneous = K @ np.array([[X_c], [Y_c], [Z_c]])

        # Extract image coordinates (u, v, d)
        u_homogeneous, v_homogeneous, d = image_homogeneous[0, 0], image_homogeneous[1, 0], image_homogeneous[2, 0]

        # Convert from homogeneous to 2D pixel coordinates
        if d != 0:
            u = u_homogeneous / d
            v = v_homogeneous / d
        else:
            u, v = None, None  # Handle the case where d is zero

        return np.array([u, v])
    
    @staticmethod
    def uv_to_world(u, v, depth, invE=None, invK=None):

        if invE is None:
            invE = Perception.get_base_invExtrinsic_matrix()
        if invK is None:
            invK = Perception.get_invIntrinsic_matrix()
        # Convert UV coordinates to homogeneous coordinates
        uv_homogeneous = np.array([[u], [v], [1.0]])

        # Convert UV to camera coordinates (assuming depth is provided)
        camera_coords = (invK @ uv_homogeneous) * depth  # Depth should be the Z_c coordinate

        # Add homogeneous coordinate
        camera_homogeneous = np.vstack((camera_coords, [1.0]))

        # Convert camera coordinates to world coordinates using the extrinsic matrix
        world_coords_homogeneous = invE @ camera_homogeneous

        # Extract world coordinates (X, Y, Z)
        X_w, Y_w, Z_w = world_coords_homogeneous[0, 0], world_coords_homogeneous[1, 0], world_coords_homogeneous[2, 0]

        return X_w, Y_w, Z_w

    @staticmethod
    def recover_homogenous_transform_pnp (image_points, world_points=AprilTag.FULL_A_LOC, K=CameraParameters.get_intrinsic_matrix()):

        distCoeffs = CameraParameters.get_distortion_coefficients()
        [_, R_exp, t] = cv2.solvePnP(world_points,
                                    image_points,
                                    K,
                                    distCoeffs,
                                    flags=cv2.SOLVEPNP_ITERATIVE)
        R, jacobian = cv2.Rodrigues(R_exp)
        
        return np.row_stack((np.column_stack((R, t)), (0, 0, 0, 1)))
    
    @staticmethod
    def transform_images(rgb_img, H = None, depth_img = None):
        """
        Transforms the RGB and depth images to a new viewpoint based on the provided camera extrinsic matrices.

        Parameters:
            rgb_img (np.array): Original RGB image.
            depth_img (np.array): Original depth image.
            K (np.array): Camera intrinsic matrix (3x3).
            T_i (np.array): Initial camera extrinsic matrix (4x4) representing the pose of the camera in the world frame.
            T_f (np.array): Final camera extrinsic matrix (4x4) representing the desired pose of the camera in the world frame.

        Returns:
            tuple: Transformed RGB image, Transformed depth image.
        """

        # Use to ensure that the entire transformed image is contained within the output
        scale_factor_w=1.
        scale_factor_h=1.
        target_size = (1280, 720)

        h, w = rgb_img.shape[:2]

        
        if H is not None: 
            H_rgb = H
        else:
            exit(0)
        
        # Create a larger canvas for RGB
        enlarged_h_rgb, enlarged_w_rgb = int(h * scale_factor_h), int(w * scale_factor_w)
        
        # Warp the RGB image using the computed homography onto the larger canvas
        warped_rgb = cv2.warpPerspective(rgb_img, H_rgb, (enlarged_w_rgb, enlarged_h_rgb), target_size)
        
        # For the depth values, we first transform them to 3D points, apply the T_relative transformation, and then project them back to depth values
        # Back-project to 3D camera coordinates
        u = np.repeat(np.arange(w)[None, :], h, axis=0)
        v = np.repeat(np.arange(h)[:, None], w, axis=1)
        
        
        
        # # DEPTH
        # Z = depth_img
        # X = (u - K[0,2]) * Z / K[0,0]
        # Y = (v - K[1,2]) * Z / K[1,1]
        
        # # Homogeneous coordinates in the camera frame
        # points_camera_frame = np.stack((X, Y, Z, np.ones_like(Z)), axis=-1)
        
        # # Apply the relative transformation to the depth points
        # points_transformed = np.dot(points_camera_frame, T_relative.T)
        
        # # Project back to depth values
        # depth_transformed = points_transformed[..., 2]
        
        # # Create a larger canvas for depth
        # enlarged_h_depth, enlarged_w_depth = int(h * scale_factor_h), int(w * scale_factor_w)
        
        # # Use the same homography as RGB for depth
        # warped_depth = cv2.warpPerspective(depth_transformed, H_rgb, (enlarged_w_depth, enlarged_h_depth), target_size)
        
        warped_depth = cv2.warpPerspective(depth_img, H_rgb, (enlarged_w_rgb, enlarged_h_rgb), target_size)
        
        return warped_rgb, warped_depth