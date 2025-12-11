import numpy as np 
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time
import math
from functions import dynamicProgram, dynamicProgramVec
import cv2
plt.close('all')

def dynamicProgramVec(unaryCosts, pairwiseCosts):
    
    # count number of positions (i.e. pixels in the scanline), and nodes at each
    # position (i.e. the number of distinct possible disparities at each position)
    nNodesPerPosition = len(unaryCosts)
    nPosition = len(unaryCosts[0])

    # define minimum cost matrix - each element will eventually contain
    # the minimum cost to reach this node from the left hand side.
    # We will update it as we move from left to right
    minimumCost = np.zeros([nNodesPerPosition, nPosition])

    # TODO: fill this function in. (hint use tiling and perform calculations columnwise with matricies)

    parents = np.zeros([nNodesPerPosition, nPosition])

    unaryCosts = np.array(unaryCosts)
    pairwiseCosts = np.array(pairwiseCosts)

    minimumCost[:, 0] = unaryCosts[:, 0]

    for c in range(1, nPosition):

        prev = minimumCost[:, c-1].reshape(-1, 1)        
        transitionCost = prev + pairwiseCosts           

        minCost = np.min(transitionCost, axis=0)        
        minInd  = np.argmin(transitionCost, axis=0)     

        minimumCost[:, c] = unaryCosts[:, c] + minCost
        parents[:, c] = minInd

    bestPath = np.zeros(nPosition, dtype=int)

    bestPath[-1] = np.argmin(minimumCost[:, -1])
    parent = bestPath[-1]

    for c in range(nPosition-2, -1, -1):
        bestPath[c] = parents[parent, c+1]
        parent = bestPath[c]

    return bestPath


def apply_undistortion(img, camera_matrix, dist_coeffs):
    """
    Apply distortion to an image based on camera parameters
    """
    h, w = img.shape[:2]
    
    # Create map for distortion
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    
    # Camera parameters
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    k1, k2, p1, p2, k3 = dist_coeffs[0], dist_coeffs[1], dist_coeffs[2], dist_coeffs[3], dist_coeffs[4]
    
    print(f"Applying distortion with k1={k1:.4f}, k2={k2:.4f}")
    
    # For each pixel in the output (distorted) image
    for v in range(h):
        for u in range(w):
            # Normalize coordinates
            x = (u - cx) / fx
            y = (v - cy) / fy
            
            # Calculate r^2
            r2 = x*x + y*y
            r4 = r2 * r2
            r6 = r4 * r2
            
            # Radial distortion
            radial = 1 + k1*r2 + k2*r4 + k3*r6
            
            # Tangential distortion
            dx = 2*p1*x*y + p2*(r2 + 2*x*x)
            dy = p1*(r2 + 2*y*y) + 2*p2*x*y
            
            # Apply distortion
            x_distorted = x * radial + dx
            y_distorted = y * radial + dy
            
            # Convert back to pixel coordinates
            map_x[v, u] = x_distorted * fx + cx
            map_y[v, u] = y_distorted * fy + cy
    
    # Apply the distortion mapping
    distorted_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    
    return distorted_img


# Camera 1 Intrinsics from your MATLAB output
camera_matrix_1 = np.array([
    [1054.9526, 0, 825.8235],
    [0, 1041.7461, 351.6964],
    [0, 0, 1]
], dtype=np.float32)

# Radial distortion for Camera 1: [k1, k2, p1, p2, k3]
dist_coeffs_1 = np.array([0.4745, -1.3516, 0, 0, 0], dtype=np.float32)

# Camera 2 Intrinsics from your MATLAB output
camera_matrix_2 = np.array([
    [1084.3657, 0, 695.6607],
    [0, 1086.9103, 347.2030],
    [0, 0, 1]
], dtype=np.float32)

# Radial distortion for Camera 2: [k1, k2, p1, p2, k3]
dist_coeffs_2 = np.array([0.0565, -0.1061, 0, 0, 0], dtype=np.float32)


# Load images
left_img_path = "/home/ramonalhk/Desktop/COMP0241/COMP0241_Lab/comp0241_25/Dataset_25/Arducam/distance/calibration_image_00_cam2.jpg"
right_img_path = "/home/ramonalhk/Desktop/COMP0241/COMP0241_Lab/comp0241_25/Dataset_25/Arducam/distance/calibration_image_00_cam1.jpg"

im1_color = cv2.imread(left_img_path)
im2_color = cv2.imread(right_img_path)
im1_original = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
im2_original = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

if im1_original is None or im2_original is None:
    raise FileNotFoundError("Images not found")

# undistortion
im1_undistorted = apply_undistortion(im1_original, camera_matrix_2, dist_coeffs_2)
im2_undistorted = apply_undistortion(im2_original, camera_matrix_1, dist_coeffs_1)

#===== DIAGNOSTIC CHECK: Show original vs distorted images =====
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(im1_color, cv2.COLOR_BGR2RGB))
plt.title('ORIGINAL - LEFT Camera (cam2)')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(im2_color, cv2.COLOR_BGR2RGB))
plt.title('ORIGINAL - RIGHT Camera (cam1)')

plt.subplot(2, 2, 3)
plt.imshow(im1_undistorted, cmap='gray')
plt.title('UNDISTORTED - LEFT Camera (cam2)')

plt.subplot(2, 2, 4)
plt.imshow(im2_undistorted, cmap='gray')
plt.title('UNDISTORTED - RIGHT Camera (cam1)')

plt.tight_layout()
plt.show()


# Use distorted images for stereo matching
im1 = im1_undistorted
im2 = im2_undistorted

# Convert to float
im1 = im1.astype(np.float32)
im2 = im2.astype(np.float32)

print("Original:", im1.shape)


imY, imX = im1.shape
print("Resized:", im1.shape)
f_pixel = (1054.9526+1084.3657)/2
#f_pixel = 1084 * scale         
baseline = 1.508              

print("Using f_pixel =", f_pixel)

maxDisp = 170
alpha = 1
noiseSD = 8

pairwiseCosts = alpha*np.ones([maxDisp, maxDisp]) - alpha*np.eye(maxDisp)
estDisp = np.zeros([imY, imX - maxDisp])

start = time.perf_counter()
for y in range(imY):
    unary = np.zeros([maxDisp, imX - maxDisp])
    for d in range(maxDisp):
        diff = im1[y, :imX-maxDisp] - im2[y, d:d+(imX-maxDisp)]
        unary[d, :] = (diff*diff) / (2*noiseSD*noiseSD)
    estDisp[y, :] = dynamicProgramVec(unary, pairwiseCosts)

print("Computing Disparity Done:", time.perf_counter()-start, "sec")

print("\n===== Disparity Info =====")
print(f"estDisp mean: {np.mean(estDisp):.2f}")
print(f"estDisp median: {np.median(estDisp):.2f}")

depth = (f_pixel * baseline) / (estDisp + 1e-6)
real_depth = 21.4

# Region around the bottle (adjust manually after viewing images)
y1, y2 = 250, min(450, estDisp.shape[0])
x1, x2 = 600, min(800, estDisp.shape[1])

roi_disparity = estDisp[y1:y2, x1:x2]

valid_disparities = roi_disparity

median_disparity = np.median(valid_disparities)
estimated_depth = (f_pixel * baseline) / median_disparity
print(f"Median disparity in ROI: {median_disparity:.2f}")
print(f"Estimated Depth: {estimated_depth:.3f}m")
print(f"Real Depth: {real_depth}m")
print(f"Error: {abs(estimated_depth - real_depth):.3f}m")



# Visualization with ROI
plt.figure()
plt.imshow(estDisp, cmap="gray")
rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
plt.gca().add_patch(rect)
plt.colorbar()
plt.title("Disparity")
plt.show()

plt.figure()
plt.imshow(depth, cmap='jet', vmin=0, vmax=30)
rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
plt.gca().add_patch(rect)
plt.colorbar()
plt.title("Depth (meters)")
plt.show()

