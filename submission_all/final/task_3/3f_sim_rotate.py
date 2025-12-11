import cv2
import numpy as np
import math
from collections import deque

# Visualization utility
def draw_filled_rect_alpha(img, top_left, bottom_right, color=(0,0,0), alpha=0.4):
    overlay = img.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def draw_text_with_box(img, text, pos, text_color=(255,255,255),
                       scale=0.9, thickness=2, box_color=(0,0,0),
                       alpha=0.5, padding=8):
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    top_left = (x - padding, y - th - padding)
    bottom_right = (x + tw + padding, y + padding)
    img[:] = draw_filled_rect_alpha(img, top_left, bottom_right, box_color, alpha)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0,0,0), thickness+3)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                text_color, thickness)


# Rotation tracker
class RotationTracker:
    def __init__(self, frame0, fps, method):
        self.method = method
        self.fps = fps
        self.frozen = False
        self.end_frame = None
        self.start_frame = None

        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        h0, w0 = gray0.shape
        self.tx = w0 - 350

        # AO deteciton with Hough Circle
        circles = cv2.HoughCircles(gray0, cv2.HOUGH_GRADIENT, 1.2, 200,
                                   param1=80, param2=30,
                                   minRadius=80, maxRadius=250)
        x0, y0, r0 = circles[0][0].astype(int)
        self.x0, self.y0, self.r0 = x0, y0, r0

        # Create AO mask
        mask = np.zeros_like(gray0)
        cv2.circle(mask, (x0, y0), r0 - 10, 255, -1)
        self.mask = mask

        # ORB baseline feature extraction
        self.orb = cv2.ORB_create(nfeatures=1800)
        self.kp0, self.des0 = self.orb.detectAndCompute(gray0, mask)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        # Match baseline to get the limit of inlier
        matches0 = self.bf.knnMatch(self.des0, self.des0, k=2)
        good0 = [m for m, n in matches0 if m.distance < 0.75*n.distance]
        self.base_inliers = len(good0)

        self.leave_thr = int(self.base_inliers * 0.60)
        self.return_thr = int(self.base_inliers * 0.60)

        # NCC baseline patch
        patch = 40
        self.patch_size = patch
        self.tmp0 = gray0[y0-patch:y0+patch, x0-patch:x0+patch]

        # Rotation cycle states
        self.left = False
        self.leave_counter = 0
        self.rotation_time = None

        self.start_frame = self._draw_start_frame(frame0)

    def _draw_start_frame(self, frame):
        img = frame.copy()
        for p in self.kp0:
            cv2.circle(img, (int(p.pt[0]), int(p.pt[1])), 3, (0,255,255), -1)
        draw_text_with_box(img, f"{self.method.upper()} Start Frame", (self.tx, 50))
        draw_text_with_box(img, f"Time: 0.00s", (self.tx, 100))
        return img

    # Compute matches, check rotation and output annotated frame in each frame
    def step(self, frame, t):

        if self.frozen:
            return self.end_frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ORB extraction
        kp, des = self.orb.detectAndCompute(gray, self.mask)
        if des is None:
            return frame

        # Matching
        matches = self.bf.knnMatch(self.des0, des, k=2)
        good = [m for m,n in matches if m.distance < 0.75*n.distance]

        # Homography and inlier
        H = None
        inliers = 0
        inlier_matches = []
        if len(good) >= 8:
            src = np.float32([self.kp0[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            H, maskH = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is not None:
                inliers = int(maskH.sum())
                inlier_matches = [good[i] for i in range(len(good)) if maskH[i] == 1]

        # Mean feature displacement
        dvals = []
        for m in good[:20]:
            (x1,y1) = self.kp0[m.queryIdx].pt
            (x2,y2) = kp[m.trainIdx].pt
            dvals.append(math.hypot(x2-x1, y2-y1))
        mean_dist = np.mean(dvals) if dvals else 0

        # NCC appearance similarity
        p = self.patch_size
        patch = gray[self.y0-p:self.y0+p, self.x0-p:self.x0+p]
        ncc = cv2.matchTemplate(patch, self.tmp0, cv2.TM_CCORR_NORMED)[0][0]

        # Departure detection
        if (mean_dist > 15) or (inliers < self.leave_thr):
            self.leave_counter += 1
        else:
            self.leave_counter = 0

        if not self.left and self.leave_counter > 8:
            self.left = True

        # Return detection
        if self.left and t > 30 and inliers > self.return_thr and ncc > 0.9 and mean_dist < 10:
            self.rotation_time = t
            self.end_frame = self._draw_end_frame(frame, inlier_matches)
            self.frozen = True
            print(f"[{self.method}] cycle detected at {t:.2f}s")
            return self.end_frame

        # Visualization
        vis = frame.copy()
        for p in kp:
            cv2.circle(vis, (int(p.pt[0]),int(p.pt[1])),2,(0,255,255),-1)
        for m in inlier_matches:
            (x1,y1)=self.kp0[m.queryIdx].pt
            (x2,y2)=kp[m.trainIdx].pt
            cv2.line(vis,(int(x1),int(y1)),(int(x2),int(y2)),(255,200,0),2)

        draw_text_with_box(vis, f"{self.method.upper()} VIEW", (20,40))
        draw_text_with_box(vis, f"Time: {t:.2f}s", (self.tx, 50))
        draw_text_with_box(vis, f"Inliers:{inliers}",(self.tx,90))
        draw_text_with_box(vis, f"MeanDist:{mean_dist:.1f}",(self.tx,130))
        return vis

    def _draw_end_frame(self, frame, matches):
        img = frame.copy()
        for m in matches:
            (x1,y1)=self.kp0[m.queryIdx].pt
            (x2,y2)=cv2.KeyPoint_convert([cv2.KeyPoint(x1,y1,1)])[0]
            cv2.line(img,(int(x1),int(y1)),(int(x2),int(y2)),(255,200,0),2)
        draw_text_with_box(img, f"{self.method.upper()} END Frame", (self.tx,50))
        draw_text_with_box(img, f"Time: {self.rotation_time:.2f}s", (self.tx,110))
        return img


# Load videos
bottom_path = ".Videos/3f/bottom.avi"
side_path   = "./Videos/3f/side.avi"

cap_b = cv2.VideoCapture(bottom_path)
cap_s = cv2.VideoCapture(side_path)

fps = cap_b.get(cv2.CAP_PROP_FPS)

ret_b, f0b = cap_b.read()
ret_s, f0s = cap_s.read()
tracker_b = RotationTracker(f0b, fps, "bottom")
tracker_s = RotationTracker(f0s, fps, "side")

frame_id = 0
done_b = False
done_s = False

# Play both videos synchronously, step through each tracker
while True:
    rb, fb = cap_b.read()
    rs, fs = cap_s.read()

    if not rb: fb = tracker_b.end_frame
    if not rs: fs = tracker_s.end_frame

    t = frame_id / fps

    vis_b = tracker_b.step(fb, t)
    vis_s = tracker_s.step(fs, t)

    if tracker_b.frozen: done_b = True
    if tracker_s.frozen: done_s = True

    vis_s = cv2.resize(vis_s, (vis_b.shape[1], vis_b.shape[0]))
    stack = np.vstack([vis_b, vis_s])

    cv2.imshow("Two-View AO Tracker", stack)

    if done_b and done_s:
        break
    if cv2.waitKey(1)==27:
        break

    frame_id += 1

cap_b.release()
cap_s.release()
cv2.destroyAllWindows()


# Same combined starting frame of side and bottom view

h1, w1 = tracker_b.start_frame.shape[:2]
start_side_resized = cv2.resize(tracker_s.start_frame, (w1, h1))
end_side_resized   = cv2.resize(tracker_s.end_frame,   (w1, h1))

start_pair = np.vstack([tracker_b.start_frame, start_side_resized])
end_pair   = np.vstack([tracker_b.end_frame,   end_side_resized])

cv2.imwrite("./Results/part_f/start_pair.png", start_pair)
cv2.imwrite("./Results/part_f/end_pair.png", end_pair)

print("\nFINISHED")
print("Bottom cycle:", tracker_b.rotation_time)
print("Side cycle:", tracker_s.rotation_time)