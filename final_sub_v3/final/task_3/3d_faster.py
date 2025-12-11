
import cv2
import numpy as np
import math
import time


# Load video
video_path = "./Videos/side.avi"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = 1.0 / fps   

ret, frame0 = cap.read()
if not ret:
    raise RuntimeError("Cannot read video")

gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
h0, w0 = gray0.shape

# AO detection with Hough Circle
circles = cv2.HoughCircles(
    gray0, cv2.HOUGH_GRADIENT, 1.2, 200,
    param1=80, param2=30, minRadius=80, maxRadius=300
)
if circles is None:
    raise RuntimeError("AO circle not detected")

x0, y0, r0 = circles[0][0].astype(int)

mask = np.zeros_like(gray0)
cv2.circle(mask, (x0, y0), r0 - 10, 255, -1)

# Baseline ORB extraction
orb = cv2.ORB_create(nfeatures=1800)
kp0, des0 = orb.detectAndCompute(gray0, mask)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Self match baseline descriptors
matches0 = bf.knnMatch(des0, des0, k=2)
good0 = [m for m,n in matches0 if m.distance < 0.75*n.distance]

BASE_INLIERS = len(good0)
print("\nBaseline inliers =", BASE_INLIERS)

INLIER_LEAVE_THR   = int(BASE_INLIERS * 0.60)
RETURN_INLIERS_THR = int(BASE_INLIERS * 0.60)

# NCC patch for appearance similarity
patch_size = 40
tmp0 = gray0[y0-patch_size:y0+patch_size, x0-patch_size:x0+patch_size]


# Rotation detection parameters
MIN_TIME = 30
MEAN_DIST_LEAVE_THR = 15
LEAVE_REQUIRED_FRAMES = 8

RETURN_MEAN_DIST_THR = 10
RETURN_NCC_THR = 0.90
RETURN_H_SCALE_THR = 0.15

# Homography check
def good_homography(H):
    if H is None: return False
    H = H / H[2,2]

    sx = (H[0,0]**2 + H[1,0]**2) ** 0.5
    sy = (H[0,1]**2 + H[1,1]**2) ** 0.5

    return ((1-RETURN_H_SCALE_THR < sx < 1+RETURN_H_SCALE_THR) and
            (1-RETURN_H_SCALE_THR < sy < 1+RETURN_H_SCALE_THR))


# Main tracking variables
has_left_start = False
leave_counter = 0
frame_id = 0
rotation_time = None

print("\nFast Tracking started...\n")

# Main processing loop
while True:

    processing_start = time.time()  # start timing for speed evaluation

    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    time_now = frame_id / fps

    # ORB feature extraction
    kp, des = orb.detectAndCompute(gray, mask)
    if des is None:
        frame_id += 1
        continue

    # Matching
    matches = bf.knnMatch(des0, des, k=2)
    good = [m for m,n in matches if m.distance < 0.75*n.distance]

    # Homography and inliers
    inliers = 0
    H = None
    if len(good) >= 8:
        src = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, maskH = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is not None:
            inliers = int(maskH.sum())

    # Mean matching dist
    dvals = []
    for m in good[:25]:
        x1,y1 = kp0[m.queryIdx].pt
        x2,y2 = kp[m.trainIdx].pt
        dvals.append(((x2-x1)**2 + (y2-y1)**2)**0.5)
    mean_dist = np.mean(dvals) if len(dvals)>0 else 0

    # NCC appearance similarity
    patch = gray[y0-patch_size:y0+patch_size, x0-patch_size:x0+patch_size]
    ncc = cv2.matchTemplate(patch, tmp0, cv2.TM_CCORR_NORMED)[0][0]

    # Departure detection
    leaving_condition = (
        mean_dist > MEAN_DIST_LEAVE_THR or
        inliers < INLIER_LEAVE_THR
    )

    if leaving_condition:
        leave_counter += 1
    else:
        leave_counter = 0

    if not has_left_start and leave_counter > LEAVE_REQUIRED_FRAMES:
        has_left_start = True
        print(f"LEFT START at t={time_now:.2f}s")

  
    # Return detection
    if (
        has_left_start and
        time_now > MIN_TIME and
        inliers > RETURN_INLIERS_THR and
        ncc > RETURN_NCC_THR and
        good_homography(H) and
        mean_dist < RETURN_MEAN_DIST_THR
    ):
        rotation_time = time_now
        print(f"CYCLE DETECTED at time={rotation_time:.2f}s")
        break

    # Record processing time for this frame
    processing_time = time.time() - processing_start
    print(f"FPS: {int(fps)} / Processing time: {processing_time:.4f}s")

    frame_id += 1

cap.release()

# Print final result
if rotation_time:
    print("\nFinal Rotation Time =", rotation_time)
else:
    print("\nNo cycle detected.")