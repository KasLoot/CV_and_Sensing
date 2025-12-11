import cv2
import numpy as np
import math
from collections import deque
import time


# Visualization utility
def draw_filled_rect_alpha(img, top_left, bottom_right, color=(0,0,0), alpha=0.4):
    overlay = img.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def draw_text_with_box(img, text, pos, text_color=(255,255,255),
                       scale=1.2, thickness=2, box_color=(0,0,0),
                       alpha=0.4, padding=10):
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    top_left = (x - padding, y - th - padding)
    bottom_right = (x + tw + padding, y + padding)

    img[:] = draw_filled_rect_alpha(img, top_left, bottom_right, box_color, alpha)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0,0,0), thickness+4)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, text_color, thickness)

# Load video 
video_path = "./Videos/bottom.avi"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

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

# Create AO mask 
mask = np.zeros_like(gray0)
cv2.circle(mask, (x0, y0), r0 - 10, 255, -1)

# ORB baseline feature extraction
orb = cv2.ORB_create(nfeatures=1800)
kp0, des0 = orb.detectAndCompute(gray0, mask)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Match baseline to get the limit of inlier
matches0 = bf.knnMatch(des0, des0, k=2)
good0 = [m for m,n in matches0 if m.distance < 0.75*n.distance]

BASE_INLIERS = len(good0)
print("\nBaseline inliers =", BASE_INLIERS)

# Set inlier threshold
INLIER_LEAVE_THR   = int(BASE_INLIERS * 0.60)
RETURN_INLIERS_THR = int(BASE_INLIERS * 0.60)

print("Leaving threshold  =", INLIER_LEAVE_THR)
print("Returning threshold =", RETURN_INLIERS_THR)


tx = w0 - 330
start_vis = frame0.copy()

# Visualize baseline ORB keypoints
for p in kp0:
    cv2.circle(start_vis, (int(p.pt[0]), int(p.pt[1])), 3, (0,255,255), -1)

draw_text_with_box(start_vis, "Start Frame", (tx,50))
draw_text_with_box(start_vis, "Time: 0.00s", (tx,110))

# NCC baseline patch
patch_size = 40
tmp0 = gray0[y0-patch_size:y0+patch_size, x0-patch_size:x0+patch_size]
start_ncc = 1.0

# Detection threshold
MIN_TIME = 30
MEAN_DIST_LEAVE_THR = 15
LEAVE_REQUIRED_FRAMES = 8

RETURN_MEAN_DIST_THR = 10    
RETURN_NCC_THR = 0.90
RETURN_H_SCALE_THR = 0.15

# Homography check
def good_homography(H):
    if H is None: 
        return False
    H = H / H[2,2]
    sx = math.sqrt(H[0,0]**2 + H[1,0]**2)
    sy = math.sqrt(H[0,1]**2 + H[1,1]**2)
    return ((1-RETURN_H_SCALE_THR < sx < 1+RETURN_H_SCALE_THR) and
            (1-RETURN_H_SCALE_THR < sy < 1+RETURN_H_SCALE_THR))


# Tracking loop variables
has_left_start = False
leave_counter = 0
frame_id = 0
rotation_time = None
end_frame_vis = None

print("\nTracking started...\n")


# Main tracking loop
while True:

    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    time_now = frame_id / fps

    # ORB detection
    kp, des = orb.detectAndCompute(gray, mask)
    if des is None:
        frame_id += 1
        continue

    # Matching with baseline
    matches = bf.knnMatch(des0, des, k=2)
    good = [m for m,n in matches if m.distance < 0.75*n.distance]

    # Homography and inliers
    inliers = 0
    inlier_matches = []
    H = None
    if len(good) >= 8:
        src = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, maskH = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        if H is not None:
            inliers = int(maskH.sum())
            inlier_matches = [good[i] for i in range(len(good)) if maskH[i]==1]

    # Mean feature displacement
    dvals = []
    for m in good[:25]:
        x1,y1 = kp0[m.queryIdx].pt
        x2,y2 = kp[m.trainIdx].pt
        dvals.append(math.hypot(x2-x1, y2-y1))
    mean_dist = np.mean(dvals) if len(dvals)>0 else 0


    # NCC appearence similarity
    patch = gray[y0-patch_size:y0+patch_size, x0-patch_size:x0+patch_size]
    ncc = cv2.matchTemplate(patch, tmp0, cv2.TM_CCORR_NORMED)[0][0]

   
    # Depature detection
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
        print(f"*** LEFT START at t={time_now:.2f}s ***")


    # Return detection
    if (
        has_left_start and
        time_now > MIN_TIME and
        inliers > RETURN_INLIERS_THR and
        ncc > RETURN_NCC_THR and
        good_homography(H) and
        mean_dist < RETURN_MEAN_DIST_THR
    ):
        print(f"[CYCLE DETECTED] t={time_now:.2f}s")
        rotation_time = time_now

        end_frame_vis = frame.copy()
        for m in inlier_matches:
            (x1,y1)=kp0[m.queryIdx].pt
            (x2,y2)=kp[m.trainIdx].pt
            cv2.line(end_frame_vis,(int(x1),int(y1)),(int(x2),int(y2)),(255,200,0),2)

        draw_text_with_box(end_frame_vis, "End Frame", (tx,50))
        draw_text_with_box(end_frame_vis, f"Time: {rotation_time:.2f}s", (tx,110))
        break

    # Visualization 
    vis = frame.copy()

    for p in kp:
        cv2.circle(vis,(int(p.pt[0]),int(p.pt[1])),2,(0,255,255),-1)

    for m in inlier_matches:
        (x1,y1)=kp0[m.queryIdx].pt
        (x2,y2)=kp[m.trainIdx].pt
        cv2.line(vis,(int(x1),int(y1)),(int(x2),int(y2)),(255,200,0),2)

    draw_text_with_box(vis, f"Time: {time_now:.2f}s", (tx,50))
    draw_text_with_box(vis, f"MeanDist: {mean_dist:.1f}", (tx,100))
    draw_text_with_box(vis, f"Inliers: {inliers}", (tx,150))
    draw_text_with_box(vis, f"ReturnDistThr: {RETURN_MEAN_DIST_THR}", (tx,200))
    draw_text_with_box(vis, f"LeftStart: {has_left_start}", (tx,250))

    cv2.imshow("AO Bottom Tracker", vis)
    end_frame_vis = vis.copy()

    frame_id += 1
    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()

# Saved start and end frame
if end_frame_vis is not None:

    h1, w1 = start_vis.shape[:2]
    h2, w2 = end_frame_vis.shape[:2]

    if w1 != w2:
        end_frame_vis = cv2.resize(end_frame_vis, (w1, int(h2 * w1 / w2)))

    combined = np.vstack([start_vis, end_frame_vis])

    save_path = "./Results/part_c/bottom/bottom_combined.png"
    cv2.imwrite(save_path, combined)
    print(f"Saved combined: {save_path}")

if rotation_time:
    print("\nFinal Rotation Time:", rotation_time)
else:
    print("\nNo cycle detected.")