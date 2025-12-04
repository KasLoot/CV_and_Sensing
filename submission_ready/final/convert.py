import cv2
import os
import numpy as np

video_path = "/Users/liuyuxin/Documents/CV_and_Sensing/submission_ready/final/Dataset_25/task_2/2b/earth_rotation.avi"
output_dir = "/Users/liuyuxin/Documents/CV_and_Sensing/submission_ready/final/Dataset_25/task_2/2b/frames"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total_frames}")

# Select 10 frames with equal spacing
# Using linspace to get 10 indices from 0 to total_frames-1
frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int)

print(f"Extracting frames at indices: {frame_indices}")

for i, frame_idx in enumerate(frame_indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        output_filename = os.path.join(output_dir, f"frame_{i:02d}.jpg")
        cv2.imwrite(output_filename, frame)
        print(f"Saved {output_filename}")
    else:
        print(f"Error reading frame at index {frame_idx}")

cap.release()
print("Done.")
