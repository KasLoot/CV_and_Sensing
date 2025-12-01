import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import deque
import cv2


from task_1 import threshold_and_region_growing


# def find_centre(image: Image.Image, threshold=30, saturation_threshold=120) -> list[int, int]:
    
#     segmented_image = threshold_and_region_growing(image, threshold=threshold, saturation_threshold=saturation_threshold)

#     # find the centre of the segmented region
#     coords = np.column_stack(np.where(segmented_image == 255))

#     if coords.shape[0] == 0:
#         print("No segmented region found.")
#         plt.imshow(segmented_image, cmap='gray')
#         plt.axis('off')
#         plt.show()
#         return

#     centre_y, centre_x = np.mean(coords, axis=0).astype(int)

#     # plot the centre on the image
#     plt.imshow(segmented_image, cmap='gray')
#     plt.scatter([centre_x], [centre_y], color='red', s=10)
#     plt.axis('off')
#     plt.show()

#     return [centre_x, centre_y]


# def find_centre_hough(image: Image.Image, threshold=30, saturation_threshold=120):
#     # 1. Get your existing segmentation
#     segmented_image = threshold_and_region_growing(image, threshold, saturation_threshold)
    
#     # Ensure it's explicitly 8-bit for OpenCV
#     if segmented_image.dtype != np.uint8:
#         segmented_image = segmented_image.astype(np.uint8)

#     # 2. Pre-processing for Hough
#     # Gaussian blur reduces noise so we don't find false circles in the texture
#     blurred = cv2.GaussianBlur(segmented_image, (9, 9), 2)
    
#     # 3. Detect Circles
#     # param1: Threshold for Canny edge detector (high threshold)
#     # param2: Accumulator threshold (The lower this is, the more false circles it detects)
#     # minRadius/maxRadius: Limit the search size to avoid finding tiny noise circles
#     circles = cv2.HoughCircles(
#         blurred, 
#         cv2.HOUGH_GRADIENT, 
#         dp=1.2,           # Inverse ratio of resolution
#         minDist=100,      # Min distance between detected centers
#         param1=50,        
#         param2=30,        # Adjust this if it misses the Earth (lower) or finds noise (higher)
#         minRadius=50,     # Approximate min radius of Earth in pixels
#         maxRadius=0       # 0 = Let it guess, or set a max limit
#     )

#     if circles is not None:
#         circles = np.round(circles[0, :]).astype("int")
        
#         # We assume the Earth is the largest circle found
#         # (x, y, radius)
#         largest_circle = max(circles, key=lambda c: c[2])
#         centre_x, centre_y, radius = largest_circle

#         # --- VISUALIZATION ---
#         plt.figure(figsize=(6,6))
#         plt.imshow(segmented_image, cmap='gray')
        
#         # Plot the center
#         plt.scatter([centre_x], [centre_y], color='red', s=50, marker='x', label='Geometric Center')
        
#         # Draw the fitted circle (to see if it correctly guessed the missing part)
#         circle_patch = plt.Circle((centre_x, centre_y), radius, color='green', fill=False, linewidth=2, label='Fitted Orbit')
#         plt.gca().add_patch(circle_patch)
        
#         plt.legend()
#         plt.axis('off')
#         plt.show()

#         return [centre_x, centre_y]

#     else:
#         print("No circle detected. Try adjusting param2 or minRadius.")
#         return None


# def fit_circle_to_outer_edge(image: Image.Image, threshold=30, saturation_threshold=120):
#     # --- STEP 1: Get the binary mask ---
#     # (Assuming you have your segmentation function)
#     # mask = threshold_and_region_growing(image, ...)
#     # For this example, let's assume 'image' IS the binary mask you showed in the screenshot
#     # If passing the original image, perform your segmentation here first.
    
#     segmented_image = threshold_and_region_growing(image, threshold, saturation_threshold)

#     # --- STEP 2: Fill the Holes (Morphological Closing) ---
#     # This connects the white patches so we get one solid "Earth" blob
#     # We use a large kernel (e.g., 15x15) to close big gaps
#     kernel = np.ones((15, 15), np.uint8)
#     closed_mask = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel)

#     # --- STEP 3: Find the Outer Contour ---
#     contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     if not contours:
#         print("No contours found.")
#         return None
    
#     # Take the largest contour (The Earth), ignoring small noise specs
#     largest_contour = max(contours, key=cv2.contourArea)
    
#     # Flatten the array to get (x, y) coordinates
#     # Shape becomes (N, 2)
#     coords = largest_contour.reshape(-1, 2)

#     # --- STEP 4: Algebraic Circle Fit (Kåsa's Method) ---
#     # This is much more robust than minEnclosingCircle for partial shapes.
#     # We solve the equation: x^2 + y^2 + Ax + By + C = 0
    
#     x = coords[:, 0]
#     y = coords[:, 1]
    
#     # Formulate the linear system: [x, y, 1] * [A, B, C] = [-x^2 - y^2]
#     A_matrix = np.column_stack((x, y, np.ones_like(x)))
#     b_vector = -(x**2 + y**2)
    
#     # Solve using Least Squares
#     solution, _, _, _ = np.linalg.lstsq(A_matrix, b_vector, rcond=None)
#     A, B, C = solution
    
#     # Calculate Center (xc, yc) and Radius (r) from coefficients
#     xc = -A / 2
#     yc = -B / 2
#     r = np.sqrt(xc**2 + yc**2 - C)

#     print(f"Estimated Center: ({xc:.2f}, {yc:.2f})")
#     print(f"Estimated Radius: {r:.2f}")

#     # --- VISUALIZATION ---
#     plt.figure(figsize=(8, 8))
#     plt.imshow(segmented_image, cmap='gray')
    
#     # Plot the calculated center
#     plt.scatter([xc], [yc], color='red', s=100, marker='+', linewidth=2, label='Geometric Center')
    
#     # Plot the fitted circle
#     circle_patch = plt.Circle((xc, yc), r, color='cyan', fill=False, linewidth=2, linestyle='--', label='Fitted Orbit')
#     plt.gca().add_patch(circle_patch)
    
#     # Plot the contour used for calculation (to verify we ignored holes)
#     plt.plot(x, y, 'g.', markersize=1, label='Contour Points')

#     plt.legend()
#     plt.title("Least Squares Circle Fit")
#     plt.axis('off')
#     plt.show()

#     return [xc, yc]


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
        
        return [cx, cy]
    else:
        print("RANSAC failed to find a valid circle.")
        return None




def track_earth_center(img1_path, img2_path, old_center):
    """
    img1_path: Path to the first image (Reference)
    img2_path: Path to the second image (Target)
    old_center: Tuple (x, y) of the Earth center in Image 1
    """
    
    # 1. Load Images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print("Error: Could not load images.")
        return

    # 2. SIFT Feature Detection
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 3. Feature Matching (using FLANN or BFMatcher)
    # FLANN is faster for large datasets, but BFMatcher is fine here.
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # 4. Apply Lowe's Ratio Test (Filter out bad matches)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # MIN_MATCH_COUNT ensures we have enough data to calculate geometry
    MIN_MATCH_COUNT = 4
    
    if len(good_matches) > MIN_MATCH_COUNT:
        # 5. Extract coordinates of the good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 6. Compute Homography Matrix (H) using RANSAC
        # RANSAC will ignore outliers (matches that don't fit the rotation model)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        print(f"Homography Matrix Calculated:\n{H}")

        # 7. Transform the specific Center Point
        # Reshape point to (1, 1, 2) for perspectiveTransform
        original_point = np.float32([old_center]).reshape(-1, 1, 2)
        
        # Apply the matrix to find the new location
        new_point = cv2.perspectiveTransform(original_point, H)
        
        new_center = (new_point[0][0][0], new_point[0][0][1])
        print(f"Old Center: {old_center}")
        print(f"New Center: {new_center}")

        # --- VISUALIZATION ---
        # Draw the matches
        matches_mask = mask.ravel().tolist()
        draw_params = dict(matchColor=(0, 255, 0), # Green matches
                           singlePointColor=None,
                           matchesMask=matches_mask, 
                           flags=2)
        
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

        # Draw the Old Center (on the left side of the stitched image)
        cv2.circle(img_matches, (int(old_center[0]), int(old_center[1])), 10, (0, 0, 255), -1) # Red Dot

        # Draw the New Center (need to offset x by width of img1 for visualization)
        offset_x = img1.shape[1]
        cv2.circle(img_matches, (int(new_center[0]) + offset_x, int(new_center[1])), 10, (255, 0, 0), -1) # Blue Dot

        plt.figure(figsize=(12, 6))
        plt.imshow(img_matches)
        plt.title(f"Red: Old Center | Blue: New Center (Projected)")
        plt.savefig('tracked_earth_center.png', bbox_inches='tight', pad_inches=0.1)
        # plt.show()
        
        return new_center

    else:
        print(f"Not enough matches found - {len(good_matches)}/{MIN_MATCH_COUNT}")
        return None

# --- USAGE EXAMPLE ---
# Replace with your actual image paths and the center coordinate you found via Region Growing
# center_from_region_growing = (300, 250) 
# new_coords = track_earth_center('earth_t0.jpg', 'earth_t1.jpg', center_from_region_growing)




def main():
    image_path = '/home/yuxin/CV_and_Sensing/final/Dataset_25/calibration_image_00_cam1.jpg'
    image = Image.open(image_path).convert('RGB')

    segmented_image = threshold_and_region_growing(image, threshold=30, saturation_threshold=120)
    old_centre = fit_circle_ransac(segmented_image)
    print(f"Old Centre: {old_centre}")

    track_earth_center('/home/yuxin/CV_and_Sensing/final/Dataset_25/calibration_image_00_cam1.jpg',
                        '/home/yuxin/CV_and_Sensing/final/Dataset_25/calibration_image_01_cam1.jpg',
                        old_centre)


if __name__ == '__main__':
    main()