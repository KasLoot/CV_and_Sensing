import cv2
import numpy as np
import math
from collections import deque


# Visualization utility
def draw_filled_rect_alpha(img, top_left, bottom_right, color=(0,0,0), alpha=0.4):
    overlay = img.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def draw_text_with_box(img, text, pos, text_color, scale=1.2, thickness=2,
                       box_color=(0,0,0), alpha=0.4, padding=10):
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)

    top_left = (x - padding, y - th - padding)
    bottom_right = (x + tw + padding, y + padding)

    img[:] = draw_filled_rect_alpha(img, top_left, bottom_right, color=box_color, alpha=alpha)

    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness + 4)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness)


# Load video
video_path = ".Videos/side_.avi"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

ret, frame0 = cap.read()
if not ret:
    raise RuntimeError("Cannot read video")

gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
h0, w0 = gray0.shape

# AO detection with Hough Circle
circles = cv2.HoughCircles(
    gray0, cv2.HOUGH_GRADIENT, 1, 200,
    param1=80, param2=30, minRadius=80, maxRadius=300
)
if circles is None:
    raise RuntimeError("No circle detected")

x0, y0, r0 = circles[0][0].astype(int)

# Create AO mask
mask = np.zeros_like(gray0)
cv2.circle(mask, (x0, y0), r0 - 10, 255, -1)

# ORB baseline features extraction
orb = cv2.ORB_create(nfeatures=1500)
kp0, des0 = orb.detectAndCompute(gray0, mask)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

#Baseline feature points visualization
start_vis = frame0.copy()
for p in kp0:
    cv2.circle(start_vis, (int(p.pt[0]), int(p.pt[1])), 3, (0,255,255), -1)

tx = w0 - 330
draw_text_with_box(start_vis, "Start Frame", (tx, 50), (255,255,0))
draw_text_with_box(start_vis, "Time: 0.00s", (tx, 110), (255,255,200))

# NCC baseline patch for appearance similarity test
patch_size = 40
tmp0 = gray0[y0-patch_size:y0+patch_size, x0-patch_size:x0+patch_size]
ncc_history = deque(maxlen=10)

# Rotation detection thresholds
MIN_TIME = 2.5
ENTER_DIFF_THR = 0.75
RETURN_NCC_THR = 0.88
RETURN_INLIERS_THR = 20

leave_counter = 0
LEAVE_REQUIRED_FRAMES = 10
has_left_start = False

frame_id = 0
rotation_time = None
end_frame_vis = None

# Homography consistency check
def is_valid_return_homography(H):
    if H is None:
        return False
    H = H / H[2, 2]
    tx = abs(H[0, 2])
    ty = abs(H[1, 2])
    if tx > 6 or ty > 6:
        return False

    sx = math.sqrt(H[0, 0]**2 + H[1, 0]**2)
    sy = math.sqrt(H[0, 1]**2 + H[1, 1]**2)
    return (0.92 < sx < 1.08) and (0.92 < sy < 1.08)


print("Tracking started...")

# Main detection loop
while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    time_now = frame_id / fps

    # ORB features extraction
    kp, des = orb.detectAndCompute(gray, mask)
    if des is None or len(kp) < 10:
        frame_id += 1
        continue

    # NCC appearance similarity
    patch = gray[y0-patch_size:y0+patch_size, x0-patch_size:x0+patch_size]
    ncc = cv2.matchTemplate(patch, tmp0, cv2.TM_CCORR_NORMED)[0][0]
    ncc_history.append(ncc)
    ncc_avg = np.mean(ncc_history)

    # Feature matching
    matches = bf.knnMatch(des0, des, k=2)
    good = [m for m,n in matches if m.distance < 0.75*n.distance]

    # Homography and inliers
    inliers = 0
    inlier_matches = []
    H = None

    if len(good) >= 8:
        src = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, mask_inlier = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is not None:
            inliers = int(mask_inlier.sum())
            inlier_matches = [good[i] for i in range(len(good)) if mask_inlier[i] == 1]

    # Departure detection
    if (ncc < ENTER_DIFF_THR or inliers < 8):
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
        ncc_avg > RETURN_NCC_THR and
        inliers > RETURN_INLIERS_THR and
        is_valid_return_homography(H)
    ):
        print(f"[CYCLE DETECTED] t={time_now:.2f}s")
        rotation_time = time_now

        end_frame_vis = frame.copy()

        for p in kp:
            cv2.circle(end_frame_vis, (int(p.pt[0]), int(p.pt[1])), 3, (0,255,255), -1)

        for m in inlier_matches:
            x1,y1 = kp0[m.queryIdx].pt
            x2,y2 = kp[m.trainIdx].pt
            cv2.line(end_frame_vis, (int(x1),int(y1)), (int(x2),int(y2)), (255,200,0), 2)

        tx = w0 - 330
        draw_text_with_box(end_frame_vis, "End Frame", (tx, 50), (255,255,0))
        draw_text_with_box(end_frame_vis, f"Time: {time_now:.2f}s", (tx, 110), (255,255,200))

        break

    # Visualization 
    vis = frame.copy()

    for p in kp:
        cv2.circle(vis, (int(p.pt[0]), int(p.pt[1])), 2, (0,255,255), -1)

    for m in inlier_matches:
        x1,y1 = kp0[m.queryIdx].pt
        x2,y2 = kp[m.trainIdx].pt
        cv2.line(vis, (int(x1),int(y1)), (int(x2),int(y2)), (255,200,0), 2)

    tx = w0 - 330
    draw_text_with_box(vis, f"Time: {time_now:.2f}s", (tx, 60), (255,255,0))
    draw_text_with_box(vis, f"NCC(avg): {ncc_avg:.2f}", (tx, 110), (200,255,200))
    draw_text_with_box(vis, f"Inliers: {inliers}", (tx, 160), (255,200,200))
    draw_text_with_box(vis, f"Left Start: {has_left_start}", (tx, 210), (255,255,255))

    cv2.imshow("AO Cycle Detector", vis)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_id += 1


cap.release()
cv2.destroyAllWindows()


# Save start and end framed

if end_frame_vis is not None:
    h1, w1 = start_vis.shape[:2]
    h2, w2 = end_frame_vis.shape[:2]

    if w1 != w2:
        end_frame_vis = cv2.resize(end_frame_vis, (w1, int(h2 * w1 / w2)))

    combined = np.vstack([start_vis, end_frame_vis])

    save_path = "./Results/part_c/side/side_combined.png"
    cv2.imwrite(save_path, combined)
    print(f"Saved combined image: {save_path}")

if rotation_time:
    print(f"\nFinal Rotation Time = {rotation_time:.2f} sec")
else:
    print("\nNo cycle detected.")