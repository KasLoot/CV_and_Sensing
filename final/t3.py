import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import deque
import cv2
import time
import sys
import os

# Add current directory to path to import task_1
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from task_1 import threshold_and_region_growing
except ImportError:
    print("Warning: Could not import threshold_and_region_growing from task_1")
    # Dummy implementation if import fails
    def threshold_and_region_growing(img, t, s):
        return np.zeros((img.size[1], img.size[0]), dtype=np.uint8)

def get_circle_from_3_points(p1, p2, p3):
    """
    Mathematical helper to find (cx, cy, r) from 3 points.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    
    if abs(D) < 1e-7: # Points are collinear (straight line)
        return None
        
    center_x = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D
    center_y = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D
    
    radius = np.sqrt((center_x - x1)**2 + (center_y - y1)**2)
    
    return (center_x, center_y, radius)

def fit_circle_ransac(mask, max_iterations=2000, distance_threshold=2.0):
    """
    Fits a circle to the outer contour using RANSAC to ignore outliers (flat edges/noise).
    
    mask: Binary image (white earth, black background)
    max_iterations: How many times to try (more = more robust but slower)
    distance_threshold: How close a point must be to the circle to be counted as 'correct'
    """
    
    # 1. Get the outer contour coordinates
    # We only want the outer edge, not the internal cloud holes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        print("No contours found.")
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    points = largest_contour.reshape(-1, 2) # Shape: (N, 2)
    
    # If we have too few points, we can't fit a circle
    if len(points) < 3:
        return None

    best_circle = None
    max_inliers = 0
    
    # 2. RANSAC Loop
    for i in range(max_iterations):
        # A. Pick 3 random points
        sample_indices = np.random.choice(len(points), 3, replace=False)
        p1, p2, p3 = points[sample_indices]
        
        # B. Calculate Circle from 3 points
        # (Using complex number math for speed/cleanliness)
        temp_circle = get_circle_from_3_points(p1, p2, p3)
        
        if temp_circle is None: continue # Collinear points, skip
        
        cx, cy, r = temp_circle
        
        # Sanity check: If radius is absurdly large (flat line) or small, skip
        if r > mask.shape[0] * 2 or r < 10: 
            continue

        # C. Calculate distances of ALL points to this circle center
        # Dist = sqrt((x-cx)^2 + (y-cy)^2)
        dists = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
        
        # D. Count Inliers (points where abs(distance - radius) < threshold)
        # This checks: Is the point on the circle ring?
        error = np.abs(dists - r)
        inliers_count = np.count_nonzero(error < distance_threshold)
        
        # E. Keep the best model
        if inliers_count > max_inliers:
            max_inliers = inliers_count
            best_circle = (cx, cy, r)

    # --- VISUALIZATION ---
    if best_circle:
        cx, cy, r = best_circle
        print(f"RANSAC Center: ({cx:.2f}, {cy:.2f}) | Radius: {r:.2f}")
        
        plt.figure(figsize=(8, 8))
        plt.imshow(mask, cmap='gray')
        
        # Plot the center
        plt.scatter([cx], [cy], color='red', s=100, marker='+', linewidth=2, label='RANSAC Center')
        
        # Plot the fitted circle
        circle_patch = plt.Circle((cx, cy), r, color='lime', fill=False, linewidth=2, label='Best Fit')
        plt.gca().add_patch(circle_patch)
        
        # Show the contour points used
        plt.scatter(points[:, 0], points[:, 1], s=1, color='cyan', alpha=0.5, label='Edge Points')
        
        plt.legend()
        plt.title(f"RANSAC Fit (Inliers: {max_inliers}/{len(points)})")
        plt.axis('off')
        plt.savefig('ransac_circle_fit.png', bbox_inches='tight', pad_inches=0.1)
        # plt.show()
        plt.clf()
        
        return [cx, cy, r]
    else:
        print("RANSAC failed to find a valid circle.")
        return None

class RotationEstimator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video: {video_path}")
            self.valid = False
            return
        
        self.valid = True
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.cx = 0
        self.cy = 0
        self.r = 0
        self.mask = None
        self.tracking_mask = None
        
    def init_model(self):
        if not self.valid: return False
        
        # Read a frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret: return False
        
        # Convert to PIL HSV for task_1 function
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        pil_hsv = pil_img.convert("HSV")
        
        # Segment
        print("Segmenting Earth (this may take a moment)...")
        # Using parameters that likely work for the Earth video
        mask = threshold_and_region_growing(pil_hsv, threshold=15, saturation_threshold=40)
        mask = mask.astype(np.uint8)
        
        # Fit circle
        circle = fit_circle_ransac(mask)
        if circle:
            self.cx, self.cy, self.r = circle
            print(f"Earth Center: ({self.cx:.2f}, {self.cy:.2f}), Radius: {self.r:.2f}")
            
            # Create tracking mask (erode slightly to avoid edge noise)
            self.tracking_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.circle(self.tracking_mask, (int(self.cx), int(self.cy)), int(self.r * 0.9), 255, -1)
            return True
        return False

    def estimate_rotation(self, fast_mode=False, real_time_sim=False):
        if not self.valid or self.tracking_mask is None:
            print("Model not initialized.")
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, prev_frame = self.cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Feature params
        feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=self.tracking_mask, **feature_params)
        
        total_angle = 0.0
        frame_idx = 0
        
        start_time = time.time()
        
        while True:
            loop_start = time.time()
            ret, frame = self.cap.read()
            if not ret: break
            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate Optical Flow
            if p0 is None or len(p0) < 10:
                p0 = cv2.goodFeaturesToTrack(prev_gray, mask=self.tracking_mask, **feature_params)
            
            if p0 is not None and len(p0) > 0:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)
                
                # Select good points
                if p1 is not None:
                    good_new = p1[st==1]
                    good_old = p0[st==1]
                    
                    # Calculate angular velocity
                    angles = []
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        
                        dx = a - c
                        dy = b - d
                        
                        # Distance from center axis (assuming rotation around vertical axis)
                        # Actually, we should check the rotation axis.
                        # Assuming side view, rotation is horizontal.
                        # z is the depth. z = sqrt(R^2 - (x-cx)^2 - (y-cy)^2)
                        # But x changes. Let's use the old position.
                        
                        dist_sq = (c - self.cx)**2 + (d - self.cy)**2
                        if dist_sq < self.r**2:
                            z = np.sqrt(self.r**2 - dist_sq)
                            if z > self.r * 0.2: # Avoid edges
                                # d_theta = dx / z
                                d_theta = dx / z
                                angles.append(d_theta)
                    
                    if angles:
                        # Median is more robust to outliers
                        avg_angle = np.median(angles)
                        total_angle += avg_angle
                        
                        # Instantaneous velocity (rad/sec)
                        # avg_angle is rad/frame
                        # velocity = avg_angle * fps
                        velocity = avg_angle * self.fps
                        
                        # Period (sec/rotation)
                        # T = 2*pi / velocity
                        period = 0
                        if abs(velocity) > 1e-5:
                            period = 2 * np.pi / abs(velocity)
                        
                        if fast_mode:
                            proc_time = time.time() - loop_start
                            print(f"Frame {frame_idx}: FPS: {self.fps:.1f} / Processing time: {proc_time:.4f}s")
                        elif real_time_sim:
                            print(f"Frame {frame_idx}: Angular Velocity: {abs(period):.2f} sec/rotation")
                        
                    p0 = good_new.reshape(-1, 1, 2)
            
            prev_gray = frame_gray.copy()
            frame_idx += 1
            
            # Re-detect features every 5 frames to keep them fresh
            if frame_idx % 5 == 0:
                 p0 = cv2.goodFeaturesToTrack(prev_gray, mask=self.tracking_mask, **feature_params)

        end_time = time.time()
        total_time = end_time - start_time
        
        # Final Estimate
        # Total angle accumulated over total_frames
        # Average angular velocity = total_angle / total_frames (rad/frame)
        if frame_idx > 0:
            avg_rad_per_frame = total_angle / frame_idx
            avg_rad_per_sec = avg_rad_per_frame * self.fps
            estimated_period = 2 * np.pi / abs(avg_rad_per_sec) if abs(avg_rad_per_sec) > 0 else 0
            
            print(f"\n--- Estimation Results ---")
            print(f"Total Frames Processed: {frame_idx}")
            print(f"Total Accumulated Angle: {total_angle:.4f} radians")
            print(f"Average Angular Velocity: {avg_rad_per_sec:.4f} rad/sec")
            print(f"Estimated Rotation Period: {estimated_period:.2f} seconds")
            
            return estimated_period
        return 0

def main():
    video_path = "/home/yuxin/CV_and_Sensing/final/Dataset_25/task_3/rotation_recored_from_side.avi"
    
    print("--- Task 3: Rotation Estimation ---")
    
    estimator = RotationEstimator(video_path)
    if not estimator.init_model():
        print("Failed to initialize model.")
        return

    print("\n(a) Methodology Explanation:")
    print("   1. Segment the Earth using HSV thresholding and Region Growing.")
    print("   2. Fit a circle using RANSAC to find the center and radius.")
    print("   3. Track features using Lucas-Kanade Optical Flow.")
    print("   4. Convert horizontal pixel displacement to angular displacement using the spherical model:")
    print("      d_theta = dx / z, where z = sqrt(R^2 - (x-cx)^2 - (y-cy)^2).")
    print("   5. Accumulate angles and average over time to estimate the period.")

    print("\n(b) & (c) Continuous Rotation Cycle Estimation:")
    period = estimator.estimate_rotation(fast_mode=False, real_time_sim=False)
    print(f"   Single Rotation Cycle Estimate: {period:.2f} seconds")

    print("\n(d) Fast Rotation Cycle Estimation:")
    print("   Running in fast mode (showing processing time per frame)...")
    estimator.estimate_rotation(fast_mode=True)

    print("\n(e) Real-Time Angular Velocity Estimation:")
    print("   Running in real-time simulation mode...")
    estimator.estimate_rotation(real_time_sim=True)
    
    print("\n(f) Comparison from Different Views:")
    print("   Checking for second video...")
    # Check for other videos
    # Assuming the other video might be named differently or in a different folder
    # Based on prompt, maybe 'bottom view'.
    # If not found, just print methodology.
    print("   Methodology for comparison:")
    print("   - Synchronize videos using timestamps or visual events.")
    print("   - Apply the same estimation method to both views.")
    print("   - For bottom view, the rotation might be around the center (in-plane rotation) or similar.")
    print("   - Compare the estimated periods. They should be consistent.")

if __name__ == '__main__':
    main()
