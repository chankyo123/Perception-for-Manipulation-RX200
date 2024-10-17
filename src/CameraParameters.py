import numpy as np


# D = [0.06480094983745523, -0.11855527086287936, 0.002104554510235678, -0.004233986289135974, 0.0]
# K = [939.3505133561931, 0.0, 649.1061836089837, 0.0, 940.1157495016945, 358.2028389997808, 0.0, 0.0, 1.0]
# R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
# P = [943.16748046875, 0.0, 643.2610607072711, 0.0, 0.0, 949.9231567382812, 358.81185251463285, 0.0, 0.0, 0.0, 1.0, 0.0]
# None
# # oST version 5.0 parameters


# [image]

# width
# 1280

# height
# 720

# [narrow_stereo]

# camera matrix
# 939.350513 0.000000 649.106184
# 0.000000 940.115750 358.202839
# 0.000000 0.000000 1.000000

# distortion
# 0.064801 -0.118555 0.002105 -0.004234 0.000000

# rectification
# 1.000000 0.000000 0.000000
# 0.000000 1.000000 0.000000
# 0.000000 0.000000 1.000000

# projection
# 943.167480 0.000000 643.261061 0.000000
# 0.000000 949.923157 358.811853 0.000000
# 0.000000 0.000000 1.000000 0.000000
class CameraParameters:
    # Camera distortion coefficients
    D = np.array([0.075,  # K1: Radial distortion coefficient
                  -0.14, # K2: Radial distortion coefficient
                  -0.002,  # P1: Tangential distortion coefficient
                  -0.005,   # P2: Tangential distortion coefficient
                  0.0])                   # K3: Radial distortion coefficient (usually zero for many models)

    # Camera intrinsic matrix
    K = np.array([[934.1, 0.0, 657.1],   # fx, 0, cx
                   [0.0, 934.1, 358.75],   # 0, fy, cy
                   [0.0, 0.0, 1.0]])                               # 0, 0, 1

    # Rectification matrix (identity matrix here)
    R = np.array([[1.0, 0.0, 0.0],   # Row 1
                  [0.0, 1.0, 0.0],   # Row 2
                  [0.0, 0.0, 1.0]])  # Row 3

    # Projection matrix
    P = np.array([[937.5, 0.0, 651.1, 0.0],  # fx, 0, cx, tx
                   [0.0, 944.9, 358.4, 0.0],  # 0, fy, cy, ty
                   [0.0, 0.0, 1.0, 0.0]])                               # 0, 0, 1, 0

    # Image dimensions
    width = 1280  # Image width in pixels
    height = 720  # Image height in pixels
    
    # D = np.array([0.1026650341946598,         # K1: Radial distortion coefficient
    #           -0.15179139294863378,      # K2: Radial distortion coefficient
    #           1.2160777224428926e-05,    # P1: Tangential distortion coefficient
    #           -0.005854123035657134,     # P2: Tangential distortion coefficient
    #           0.0])                       # K3: Radial distortion coefficient (usually zero)

    # K = np.array([[921.3392567018761, 0.0, 636.5103580035612],    # fx, 0, cx
    #             [0.0, 926.7790122711867, 350.15254232590337],  # 0, fy, cy
    #             [0.0, 0.0, 1.0]])                               # 0, 0, 1

    # R = np.array([[1.0, 0.0, 0.0],   # Row 1
    #             [0.0, 1.0, 0.0],   # Row 2
    #             [0.0, 0.0, 1.0]])  # Row 3

    # P = np.array([[933.7493896484375, 0.0, 628.008144523119, 0.0],  # fx, 0, cx, tx
    #             [0.0, 946.5015869140625, 349.623492509505, 0.0],  # 0, fy, cy, ty
    #             [0.0, 0.0, 1.0, 0.0]])                               # 0, 0, 1, 0

    # # Image dimensions
    # width = 1280  # Image width in pixels
    # height = 720  # Image height in pixels

    @staticmethod
    def get_distortion_coefficients():
        return CameraParameters.D

    @staticmethod
    def get_intrinsic_matrix():
        return CameraParameters.K

    @staticmethod
    def get_rectification_matrix():
        return CameraParameters.R

    @staticmethod
    def get_projection_matrix():
        return CameraParameters.P

    @staticmethod
    def get_image_dimensions():
        return CameraParameters.width, CameraParameters.height
    
    
    