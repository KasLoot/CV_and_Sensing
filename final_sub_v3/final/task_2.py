import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import time
import sys

def crop_with_mask(img, mask, box):
    (x1, y1, x2, y2) = box

    # Crop image + crop mask
    cropped = img[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]

    # White background outside the mask
    cropped_white = cropped.copy()
    cropped_white[mask_crop == 0] = [255, 255, 255]

    # Place back into white canvas
    canvas = np.ones_like(img) * 255
    canvas[y1:y2, x1:x2] = cropped_white

    return canvas

def hough_circle_mask(gray_image, param1=50, param2=30, min_radius=300, max_radius=500):

    h, w = gray_image.shape

    circles = cv2.HoughCircles(
        gray_image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=h,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    # Create mask from detected circles
    mask = np.zeros_like(gray_image)
    box = (0, 0, w, h)  # Default box in case no circles are found
    
    first_circle = circles[0][0] if circles is not None else None
    if first_circle is not None:
        x, y, r = map(int, first_circle)
        cv2.circle(mask, (x, y), r, 255, thickness=-1)
        box = (max(x - r, 0), max(y - r, 0), min(x + r, w), min(y + r, h))
    
    if first_circle is None:
        # print("No circles detected.")
        return mask

    return mask


def find_centre(binary_mask):
    """Find the centre of the largest connected component in the binary mask."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return None  # No components found
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    centre = tuple(map(int, centroids[largest_label]))
    return centre

def video_to_frames(video_dir, sample_rate=100):
    
    # sample and save frames from video
    frames = []
    cap = cv2.VideoCapture(video_dir)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if count % sample_rate == 0:
                frames.append(frame)
            count += 1
        else:
            break
    cap.release()
    print(f"Total frames extracted: {len(frames)}")

    # save frames to disk
    os.makedirs('./Dataset_25/task_2/extracted_frames', exist_ok=True)
    for i, frame in tqdm(enumerate(frames), desc="Saving frames", total=len(frames)):
        cv2.imwrite(os.path.join('./Dataset_25/task_2/extracted_frames', f'frame_{i:04d}.png'), frame)


def get_images_from_video(video_dir, sample_rate=100) -> list[np.ndarray]:
    bgr_images = []
    gray_images = []
    hsv_images = []

    cap = cv2.VideoCapture(video_dir)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if count % sample_rate == 0:
                bgr_images.append(frame)
                gray_images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                hsv_images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
            count += 1
        else:
            break
    cap.release()
    
    print(f"Total frames extracted: {len(bgr_images)}")

    return bgr_images, gray_images, hsv_images

def get_images_from_dir(image_dir) -> list[np.ndarray]:
    bgr_images = []
    gray_images = []
    hsv_images = []
    for filename in os.listdir(image_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(image_dir, filename)
            image = cv2.imread(img_path)
            bgr_images.append(image)
            gray_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            hsv_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    return bgr_images, gray_images, hsv_images


def measure_swing(sample_rate=100, save_dir='./results/task_2'):
    images_dir = './Dataset_25/task_2/extracted_frames'
    bgr_images, gray_images, hsv_images = get_images_from_dir(images_dir)

    centres = []
    lower_bound=np.array([55, 50, 150])
    upper_bound=np.array([150, 255, 255])
    for i in tqdm(range(len(hsv_images)), desc="Processing images"):
        # mask = colour_thresholding_hough_circle(hsv_images[i], gray_image=gray_images[i], lower_bound=lower_bound, upper_bound=upper_bound, param2=30, min_radius=200, max_radius=500)
        mask = hough_circle_mask(gray_images[i], param2=30, min_radius=100, max_radius=250)

        # plt.subplot(1, 3, 1)
        # plt.title('Original Image')
        # plt.imshow(cv2.cvtColor(bgr_images[i], cv2.COLOR_BGR2RGB))

        # plt.subplot(1, 3, 2)
        # plt.title('Mask')
        # plt.imshow(mask, cmap='gray')
        # plt.axis('off')

        # cropped = crop_with_mask(bgr_images[i], mask, (0, 0, bgr_images[i].shape[1], bgr_images[i].shape[0]))
        # plt.subplot(1, 3, 3)
        # plt.title('Cropped with Mask')
        # plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        # plt.axis('off')

        # plt.show()

        centre = find_centre(mask)
        centres.append(centre)

    # Find first valid centre to use as reference
    ref_image_index = 0
    for idx, c in enumerate(centres):
        if c is not None:
            ref_image_index = idx
            break
            
    ref_image = bgr_images[ref_image_index].copy()
    ref_centre = centres[ref_image_index]
    
    distances = []
    x_displacements = []
    y_displacements = []
    frames = []

    # plot the centeres on the reference image
    for i, centre in enumerate(centres):
        if centre is not None and ref_centre is not None:
            cv2.circle(ref_image, centre, 5, (0, 0, 255), -1)
            # cv2.line(ref_image, ref_centre, centre, (255, 0, 0), 2)
            
            # Calculate displacement from reference center
            dx = centre[0] - ref_centre[0]
            dy = centre[1] - ref_centre[1]
            dist = np.sqrt(dx**2 + dy**2)
            
            distances.append(dist)
            x_displacements.append(dx)
            y_displacements.append(dy)
            frames.append(i)
    

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB))
    plt.title('Center Trajectory')
    plt.suptitle(f'Sample Rate: {1} in {sample_rate} frames')
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, 'task_2_centres_swing.png'))
    plt.show()
    
    # Total Displacement
    plt.figure(figsize=(10, 6))
    plt.plot(frames, distances, label='Total Displacement', color='blue')
    plt.title('Total Displacement from Start')
    plt.xlabel('Frame Index')
    plt.ylabel('Distance (pixels)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'task_2_movement_total_displacement.png'))
    plt.show()

    # X Displacement
    plt.figure(figsize=(10, 6))
    plt.plot(frames, x_displacements, label='X Displacement', color='green')
    plt.title('X Displacement from Start')
    plt.xlabel('Frame Index')
    plt.ylabel('X Offset (pixels)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'task_2_movement_x_displacement.png'))
    plt.show()

    # Y Displacement
    plt.figure(figsize=(10, 6))
    plt.plot(frames, y_displacements, label='Y Displacement', color='red')
    plt.title('Y Displacement from Start')
    plt.xlabel('Frame Index')
    plt.ylabel('Y Offset (pixels)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'task_2_movement_y_displacement.png'))
    plt.show()


def measure_distance():
    #WORK GEH VERSION 2c
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


    def apply_distortion(img, camera_matrix, dist_coeffs):
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
    left_img_path = "./Dataset_25/task_2/calibration_image_00_cam2.jpg"
    right_img_path = "./Dataset_25/task_2/calibration_image_00_cam1.jpg"

    im1_color = cv2.imread(left_img_path)
    im2_color = cv2.imread(right_img_path)
    im1_original = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    im2_original = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

    if im1_original is None or im2_original is None:
        raise FileNotFoundError("Images not found")

    # ===== APPLY DISTORTION HERE =====
    print("\n===== Applying Distortion =====")
    print("Distorting Camera 2 (left image)...")
    im1_distorted = apply_distortion(im1_original, camera_matrix_2, dist_coeffs_2)

    print("Distorting Camera 1 (right image)...")
    im2_distorted = apply_distortion(im2_original, camera_matrix_1, dist_coeffs_1)

    # Save distorted images
    cv2.imwrite(os.path.join(save_dir, 'c/task_2c_distorted_cam2_left.jpg'), im1_distorted)
    cv2.imwrite(os.path.join(save_dir, 'c/task_2c_distorted_cam1_right.jpg'), im2_distorted)

    # ===== DIAGNOSTIC CHECK: Show original vs distorted images =====
    print("\n===== Camera Setup Check =====")
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(im1_color, cv2.COLOR_BGR2RGB))
    plt.title('ORIGINAL - LEFT Camera (cam2)')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(im2_color, cv2.COLOR_BGR2RGB))
    plt.title('ORIGINAL - RIGHT Camera (cam1)')

    plt.subplot(2, 2, 3)
    plt.imshow(im1_distorted, cmap='gray')
    plt.title('DISTORTED - LEFT Camera (cam2)')

    plt.subplot(2, 2, 4)
    plt.imshow(im2_distorted, cmap='gray')
    plt.title('DISTORTED - RIGHT Camera (cam1)')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'c/task_2c_distortion_check.png'))
    plt.show()



    # Use distorted images for stereo matching
    im1 = im1_distorted
    im2 = im2_distorted

    # Convert to float
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    print("Original:", im1.shape)

    # downsample, because DP stereo cannot run full resolution
    scale = 0.35
    im1 = cv2.resize(im1, (0,0), fx=scale, fy=scale)
    im2 = cv2.resize(im2, (0,0), fx=scale, fy=scale)

    imY, imX = im1.shape
    print("Resized:", im1.shape)
    f_pixel = (1054.9526+1084.3657)/2*scale
    #f_pixel = 1084 * scale         
    baseline = 1.508              

    print("Using f_pixel =", f_pixel)

    maxDisp = 80
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
    print(f"estDisp shape: {estDisp.shape}")
    print(f"estDisp min/max: {estDisp.min():.2f} / {estDisp.max():.2f}")
    print(f"estDisp mean: {np.mean(estDisp):.2f}")
    print(f"estDisp median: {np.median(estDisp):.2f}")
    print(f"Number of zeros: {np.sum(estDisp == 0)}")

    depth = (f_pixel * baseline) / (estDisp + 1e-6)
    real_depth = 21.4

    # Region around the bottle (adjust manually after viewing images)
    y1, y2 = int(250*scale), min(int(450*scale), estDisp.shape[0])
    x1, x2 = int(650*scale), min(int(750*scale), estDisp.shape[1])

    print(f"\n===== ROI Coordinates =====")
    print(f"Scaled ROI: y[{y1}:{y2}], x[{x1}:{x2}]")
    roi_disparity = estDisp[y1:y2, x1:x2]

    valid_disparities = roi_disparity

    median_disparity = np.median(valid_disparities)
    estimated_depth = (f_pixel * baseline) / median_disparity
    print(f"Median disparity in ROI: {median_disparity:.2f}")
    print(f"Estimated Depth: {estimated_depth:.3f}m")
    print(f"Real Depth: {real_depth}m")
    print(f"Error: {abs(estimated_depth - real_depth):.3f}m")



    # Visualization
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(im1, cmap='gray')
    plt.title('Left Image (distorted & processed)')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(estDisp, cmap='jet')
    plt.title('Disparity Map')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(depth, cmap='jet', vmin=0, vmax=30)
    plt.title('Depth Map (0-30m)')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'c/task_2c_distance_disparity_depth.png'))
    plt.show()

    # Individual plots
    plt.figure()
    plt.imshow(estDisp, cmap="gray")
    plt.colorbar()
    plt.title("Disparity")
    plt.savefig(os.path.join(save_dir, 'c/task_2c_distance_disparity.png'))
    plt.show()

    plt.figure()
    plt.imshow(depth, cmap='jet', vmin=0, vmax=30)
    plt.colorbar()
    plt.title("Depth (meters)")
    plt.savefig(os.path.join(save_dir, 'c/task_2c_distance_depth.png'))
    plt.show()

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def find_centre_demo():
    bgr_image = cv2.imread('./Dataset_25/task_2/extracted_frames/frame_0000.png')
    gray_image = cv2.imread('./Dataset_25/task_2/extracted_frames/frame_0000.png', cv2.IMREAD_GRAYSCALE)
    mask = hough_circle_mask(gray_image, param2=30, min_radius=100, max_radius=250)

    centre = find_centre(mask)
    print(f"Detected Centre: {centre}")

    # Visualize
    color_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    if centre is not None:
        cv2.circle(color_image, centre, 5, (255, 0, 0), -1)

    plt.imshow(color_image)
    plt.title('Detected Centre')
    plt.axis('off')
    plt.savefig('./results/task_2/detected_centre_demo.png')
    plt.show()


if __name__ == "__main__":
    video_path = './Dataset_25/task_2/6f.avi'
    sample_rate = 100
    save_dir = './results/task_2'
    # video_to_frames(video_path, sample_rate=sample_rate)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'c'), exist_ok=True)

    find_centre_demo()

    measure_swing(sample_rate=sample_rate, save_dir=save_dir)
    sys.stdout = Logger(os.path.join(save_dir, 'c/output.txt'))

    measure_distance()