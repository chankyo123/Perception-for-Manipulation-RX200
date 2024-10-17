import pyrealsense2 as rs
import numpy as np
import cv2

# Configure the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable the color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a new set of frames
        frames = pipeline.wait_for_frames()

        # Get the color frame
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue
        
        # Convert the color frame to a NumPy array
        color_image = np.asanyarray(color_frame.get_data())

        # Display the color image using OpenCV
        cv2.imshow('RealSense Color Frame', color_image)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline and close any open windows
    pipeline.stop()
    cv2.destroyAllWindows()
