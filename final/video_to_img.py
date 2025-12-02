import cv2
import numpy as np
from PIL import Image
import time

def read_avi_with_pillow(video_path):
    """Reads an AVI video frame by frame and converts to Pillow images."""
    
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 0
    while True:
        # Read a new frame
        ret, frame = cap.read()
        
        # If frame is read correctly 'ret' is True
        if not ret:
            break
        
        # OpenCV reads in BGR format, Pillow uses RGB
        # Convert BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the NumPy array (the frame) to a Pillow Image object
        pil_img = Image.fromarray(rgb_frame)
        
        # --- You can now use Pillow methods on 'pil_img' ---
        # Example: Display or save the frame
        # pil_img.show() 
        # pil_img.save(f"frame_{frame_count:04d}.png")

        print(f"Processed frame {frame_count}, size: {pil_img.size}")
        frame_count += 1
        # Optional: Add a small delay if displaying many images quickly
        # time.sleep(0.01)

    # Release the capture when done
    cap.release()
    cv2.destroyAllWindows()

# Replace 'your_video.avi' with the path to your AVI file
read_avi_with_pillow('/Users/liuyuxin/Documents/CV_and_Sensing/final/Dataset_25/earth_rotation.avi')
