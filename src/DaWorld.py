import numpy as np 

class DaWorld:
    cam_pitch = -173 * np.pi / 180  # Convert degrees to radians

    # CAM TO WORLD
    R = np.array([[1, 0, 0],
                  [0, np.cos(cam_pitch), -np.sin(cam_pitch)],
                  [0, np.sin(cam_pitch), np.cos(cam_pitch)]])
    
    SCALED_TAGS = np.array([
        [320, 553.85] ,
        [960, 553.85],  # Transformed P1
        [960, 221.54],   # Transformed P2
        [320, 221.54],    # Transformed P3
       # Transformed P4
])

    TARGET_CAM_ORIENTATION = np.array([
                        [1, 0,  0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 1000],   
                        [0, 0,  0, 1] ])

    t = np.array([[0], [250], [996]])
    
    @staticmethod
    def get_cam_to_world_rotation():
        return DaWorld.R
    
    @staticmethod
    def get_cam_to_world_translation():
        return DaWorld.t
    
    
    # 250 400, 750 400, 750 200, 250 200, be