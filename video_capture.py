import cv2
import os
from datetime import datetime
import time

# Parameters
output_folder = "./captured_videos"  # Folder to save the videos
start_stop_key = 'r'  # Key to start/stop recording (r for record)
exit_key = 'q'  # Key to exit the program

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the webcam (use the appropriate index if multiple cameras are connected)
cap = cv2.VideoCapture(2)  # Change the index to 0, 1, 2, etc., if needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

# Get actual frame dimensions and FPS from camera
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
camera_fps = cap.get(cv2.CAP_PROP_FPS)

# Use camera's actual FPS, or default to 30 if it returns 0
if camera_fps == 0 or camera_fps > 60:
    camera_fps = 30.0
    print(f"Camera FPS not detected, using default: {camera_fps}")
else:
    print(f"Camera FPS detected: {camera_fps}")

print(f"Camera resolution: {frame_width}x{frame_height}")
print(f"Press '{start_stop_key}' to start/stop recording.")
print(f"Press '{exit_key}' to quit.")

recording = False
out = None
video_count = 0
frame_count = 0
start_time = time.time()

while True:
    loop_start = time.time()
    
    # Read a frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Unable to read from the camera.")
        break
    
    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    processing_time = time.time() - loop_start
    
    # Add recording indicator and FPS info to the frame
    if recording:
        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)  # Red circle
        cv2.putText(frame, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 255), 2)
    
    # Display FPS and processing time
    cv2.putText(frame, f"FPS: {current_fps:.1f} / Processing: {processing_time:.3f}s", 
                (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow("Video Capture", frame)
    
    # Write frame if recording
    if recording and out is not None:
        out.write(frame)
    
    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(start_stop_key):  # Start/Stop recording
        if not recording:
            # Start recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(output_folder, f"video_{video_count:02d}_{timestamp}.avi")
            
            # Use MJPG codec with camera's actual FPS
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            
            # IMPORTANT: Use the camera's actual FPS for recording
            out = cv2.VideoWriter(video_path, fourcc, camera_fps, (frame_width, frame_height))
            
            if not out.isOpened():
                print("Error: Unable to create video writer. Trying alternative codec...")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_path = os.path.join(output_folder, f"video_{video_count:02d}_{timestamp}.avi")
                out = cv2.VideoWriter(video_path, fourcc, camera_fps, (frame_width, frame_height))
            
            if out.isOpened():
                recording = True
                print(f"Recording started: {video_path}")
                print(f"Recording at {camera_fps} FPS")
                # Reset frame counter for recording FPS
                frame_count = 0