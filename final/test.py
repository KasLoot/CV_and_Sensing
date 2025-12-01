import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import deque
import cv2

def threshoding(image: Image.Image, threshold: list):
    copy_image = image.copy()
    # apply gaussian blur
    blurred_image = copy_image.filter(ImageFilter.GaussianBlur(radius=2))

    hsv_img = blurred_image.convert("HSV")
    hsv_array = np.array(hsv_img)

    hue_channel = hsv_array[:, :, 0]
    mask = (hue_channel >= threshold[0]) & (hue_channel <= threshold[1])
    binary_image = np.ones_like(hue_channel)*255
    binary_image[mask] = 0
    # plt.imshow(binary_image, cmap='gray')
    # plt.axis('off')
    # plt.show()

    return binary_image



def region_growing(blurred_image: Image.Image, seed_point: tuple, threshold: int):
# 1. Convert to HSV and extract Hue
    # Note: convert returns a new image, so .copy() is not strictly necessary
    hsv_img = blurred_image.convert("HSV")
    hsv_array = np.array(hsv_img)
    hue_channel = hsv_array[:, :, 0].astype(int) # Use int to prevent overflow during math

    height, width = hue_channel.shape
    seed_x, seed_y = seed_point
    
    # Validation to ensure seed is within bounds
    if not (0 <= seed_x < width and 0 <= seed_y < height):
        raise ValueError("Seed point is out of image bounds")

    seed_value = hue_channel[seed_y, seed_x]
    
    # 2. Setup Output and Tracking
    binary_image = np.zeros((height, width), dtype=np.uint8)
    
    # Use a boolean mask for 'visited' - much faster than a set of tuples
    visited_mask = np.zeros((height, width), dtype=bool)
    
    # Use deque for efficient pop/append (Breadth-First Search)
    to_visit = deque([(seed_x, seed_y)])
    visited_mask[seed_y, seed_x] = True

    # 3. Processing Loop
    while to_visit:
        x, y = to_visit.popleft() # BFS (Use pop() for DFS)
        
        # Mark pixel in binary output
        binary_image[y, x] = 255

        # Check 8-connected neighbors
        # (Defining neighbors explicitly is faster than nested loops in Python)
        neighbors = [
            (x-1, y), (x+1, y), (x, y-1), (x, y+1), # 4-connected
            (x-1, y-1), (x+1, y-1), (x-1, y+1), (x+1, y+1) # Diagonals
        ]

        for nx, ny in neighbors:
            # Check bounds and if already visited
            if 0 <= nx < width and 0 <= ny < height and not visited_mask[ny, nx]:
                
                curr_val = hue_channel[ny, nx]
                
                # 4. FIX: Circular Hue Distance Calculation
                # Distance is the minimum of direct difference or wrap-around difference
                diff = abs(curr_val - seed_value)
                circular_diff = min(diff, 255 - diff)
                
                if circular_diff <= threshold:
                    visited_mask[ny, nx] = True # Mark as visited immediately when adding to queue
                    to_visit.append((nx, ny))

    return binary_image



def draw_roc_curve(image: Image.Image, ground_truth_image: Image.Image):
    gt_array = np.array(ground_truth_image)

    true_positive_thresholding = []
    true_negative_thresholding = []
    false_positive_thresholding = []
    false_negative_thresholding = []
    thresholds = range(0, 256, 10)

    for thresh in tqdm(thresholds):
        binary_image = threshoding(image, threshold=[0, thresh])
        tp = np.sum((binary_image == 255) & (gt_array == 255))
        tn = np.sum((binary_image == 0) & (gt_array == 0))
        fp = np.sum((binary_image == 255) & (gt_array == 0))
        fn = np.sum((binary_image == 0) & (gt_array == 255))
        true_positive_thresholding.append(tp)
        false_positive_thresholding.append(fp)
        true_negative_thresholding.append(tn)
        false_negative_thresholding.append(fn)
    
    true_positive_rate_thresholding = np.array(true_positive_thresholding) / (np.array(true_positive_thresholding) + np.array(false_negative_thresholding) + 1e-6)
    false_positive_rate_thresholding = np.array(false_positive_thresholding) / (np.array(false_positive_thresholding) + np.array(true_negative_thresholding) + 1e-6)

    plt.plot(false_positive_rate_thresholding, true_positive_rate_thresholding, marker='o')
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.title('ROC Curve')
    plt.grid()
    plt.savefig('threshoding_roc_curve_test.png')
    # plt.show()
    print("ROC curve saved as 'threshoding_roc_curve_test.png'")
    plt.clf()

    tp_region_growing = []
    tn_region_growing = []
    fp_region_growing = []
    fn_region_growing = []
    thresholds = range(0, 256, 10)
    seed_point = (500, 500)  # Example seed point
    blurred_image = image.copy().filter(ImageFilter.GaussianBlur(radius=2))
    hsv_img = blurred_image.convert("HSV")
    for thresh in tqdm(thresholds):
        binary_image = region_growing_2(hsv_img, seed_point=seed_point, threshold=thresh)
        tp = np.sum((binary_image == 255) & (gt_array == 255))
        tn = np.sum((binary_image == 0) & (gt_array == 0))
        fp = np.sum((binary_image == 255) & (gt_array == 0))
        fn = np.sum((binary_image == 0) & (gt_array == 255))
        tp_region_growing.append(tp)
        fp_region_growing.append(fp)
        tn_region_growing.append(tn)
        fn_region_growing.append(fn)
    
    true_positive_rate_region_growing = np.array(tp_region_growing) / (np.array(tp_region_growing) + np.array(fn_region_growing) + 1e-6)
    false_positive_rate_region_growing = np.array(fp_region_growing) / (np.array(fp_region_growing) + np.array(tn_region_growing) + 1e-6)

    plt.plot(false_positive_rate_region_growing, true_positive_rate_region_growing, marker='o', color='red')
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.title('ROC Curve for Region Growing')
    plt.grid()
    plt.savefig('region_growing_roc_curve_test.png')
    # plt.show()
    print("ROC curve saved as 'region_growing_roc_curve_test.png'")
    plt.clf()

    tp_combined = []
    tn_combined = []
    fp_combined = []
    fn_combined = []
    thresholds = range(0, 256, 10)
    blurred_image = image.copy().filter(ImageFilter.GaussianBlur(radius=2))
    hsv_img = blurred_image.convert("HSV")
    for thresh in tqdm(thresholds):
        binary_image_combined = threshold_and_region_growing(hsv_img, threshold=thresh)
        tp = np.sum((binary_image_combined == 255) & (gt_array == 255))
        tn = np.sum((binary_image_combined == 0) & (gt_array == 0))
        fp = np.sum((binary_image_combined == 255) & (gt_array == 0))
        fn = np.sum((binary_image_combined == 0) & (gt_array == 255))
        tp_combined.append(tp)
        fp_combined.append(fp)
        tn_combined.append(tn)
        fn_combined.append(fn)

    true_positive_rate_combined = np.array(tp_combined) / (np.array(tp_combined) + np.array(fn_combined) + 1e-6)
    false_positive_rate_combined = np.array(fp_combined) / (np.array(fp_combined) + np.array(tn_combined) + 1e-6)
    plt.plot(false_positive_rate_combined, true_positive_rate_combined, marker='o', color='green')
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.title('ROC Curve for Combined Method')
    plt.grid()
    plt.savefig('combined_roc_curve_test.png')
    # plt.show()
    print("ROC curve saved as 'combined_roc_curve_test.png'")
    plt.clf()

    # plotting both curves together
    plt.plot(false_positive_rate_thresholding, true_positive_rate_thresholding, marker='.', color='blue', label='Thresholding')
    plt.plot(false_positive_rate_region_growing, true_positive_rate_region_growing, marker='.', color='red', label='Region Growing')
    plt.plot(false_positive_rate_combined, true_positive_rate_combined, marker='.', color='green', label='Combined Method')
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.grid()
    plt.savefig('roc_curve_comparison_test.png')
    plt.show()
    print("ROC curve comparison saved as 'roc_curve_comparison_test.png'")


def get_safe_ocean_seeds(hsv_img: Image.Image, saturation_threshold: int = 120):
    hsv_img_copy = hsv_img.copy()
    
    hsv_array = np.array(hsv_img_copy)
    
    H = hsv_array[:, :, 0]
    S = hsv_array[:, :, 1]
    V = hsv_array[:, :, 2]

    # --- DEFINE THE SAFETY ZONES ---
    
    # 1. HUE: Target BLUE only. 
    # In PIL (0-255 scale), Blue is approx 170. 
    # Let's target a range of 140 to 190 to be safe.
    # This strictly avoids Yellow (approx 40) and Green (approx 85).
    blue_mask = (H > 140) & (H < 190)

    # 2. SATURATION: Avoid White Clouds and Lights.
    # Clouds and Lights have very low saturation (near 0).
    # Deep Ocean has high saturation.
    # We require Saturation > 50 to avoid picking up white lights as seeds.
    sat_mask = (S > saturation_threshold)

    # Combine them
    final_seed_mask = blue_mask & sat_mask

    # Create Binary Image (0 for background, 255 for seed)
    binary_seeds = np.zeros_like(H)
    binary_seeds[final_seed_mask] = 255

    return binary_seeds


def threshold_and_region_growing(hsv_image: Image.Image, threshold: int, saturation_threshold: int = 120):
    binary_seeds = get_safe_ocean_seeds(hsv_image, saturation_threshold=saturation_threshold)
    # plt.imshow(binary_seeds, cmap='gray')
    # plt.axis('off')
    # plt.show()

    segmented_images = []
    for y in range(0, binary_seeds.shape[0], 50):
        for x in range(0, binary_seeds.shape[1], 50):
            if binary_seeds[y, x] == 255:
                # print(f"Seed found at: ({x}, {y})")
                segmented_image = region_growing_2(hsv_image, seed_point=(x, y), threshold=threshold)
                segmented_images.append(segmented_image)
    # Combine all segmented images
    combined_segmented_image = np.zeros_like(binary_seeds)
    for seg_img in segmented_images:
        combined_segmented_image = np.maximum(combined_segmented_image, seg_img)
    # plt.imshow(combined_segmented_image, cmap='gray')
    # plt.axis('off')
    # plt.show()

    return combined_segmented_image


def region_growing_2(hsv_image: Image.Image, seed_point: tuple, threshold: int):
    # Setup
    hsv_image_copy = hsv_image.copy()
    hsv_array = np.array(hsv_image_copy)
    hue_channel = hsv_array[:, :, 0]

    seeds = seed_point # Your ocean seed
    connectivity = 4 # or 8
    flags = connectivity | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE
    
    # floodFill expects a numpy array. 
    # We work on the Hue channel directly.
    # Note: cv2.floodFill modifies the image in-place, so we copy.
    work_image = hue_channel.copy()
    
    h, w = work_image.shape
    mask = np.zeros((h+2, w+2), np.uint8) # floodFill requires mask to be 2px larger
    
    # Run Region Growing (Instant in C++)
    # For 1-channel image, newVal is a scalar, loDiff/upDiff are scalars.
    cv2.floodFill(work_image, mask, seeds, 255, (threshold,), (threshold,), flags)
    
    # Extract the actual binary map (Crop the 2px border)
    binary_output = mask[1:-1, 1:-1]

    return binary_output


def YoudensJ_evaluation(hsv_img: Image.Image, ground_truth_image: Image.Image):
    gt_array = np.array(ground_truth_image)

    saturations = range(0, 256, 10)
    thresholds = range(0, 256, 10)
    
    heatmap_data = np.zeros((len(saturations), len(thresholds)))

    best_jouden_index = -1
    best_saturation = -1
    best_threshold = -1
    
    for i, sat in tqdm(enumerate(saturations), total=len(saturations)):
        for j, thresh in enumerate(thresholds):
            binary_image = threshold_and_region_growing(hsv_img, threshold=thresh, saturation_threshold=sat)
            tp = np.sum((binary_image == 255) & (gt_array == 255))
            tn = np.sum((binary_image == 0) & (gt_array == 0))
            fp = np.sum((binary_image == 255) & (gt_array == 0))
            fn = np.sum((binary_image == 0) & (gt_array == 255))

            tpr = tp / (tp + fn + 1e-6)
            fpr = fp / (fp + tn + 1e-6)

            jouden_index = tpr - fpr
            heatmap_data[i, j] = jouden_index

            if jouden_index > best_jouden_index:
                best_jouden_index = jouden_index
                best_saturation = sat
                best_threshold = thresh
    
    print(f"Best Youden's J Index: {best_jouden_index} at Saturation: {best_saturation}, Threshold: {best_threshold}")

    plt.figure(figsize=(10, 8))
    # extent=[x_min, x_max, y_max, y_min] to match image coordinates (y going down)
    # x is threshold, y is saturation
    plt.imshow(heatmap_data, extent=[0, 255, 255, 0], aspect='auto', cmap='viridis')
    plt.colorbar(label="Youden's J Index")
    plt.xlabel('Threshold')
    plt.ylabel('Saturation')
    plt.title("Youden's J Index Heatmap")
    plt.savefig('youdens_j_heatmap_test.png')
    print("Heatmap saved as 'youdens_j_heatmap_test.png'")
    plt.clf()

    return best_saturation, best_threshold
    


def main():
    image = Image.open("/home/yuxin/CV_and_Sensing/final/Dataset_25/Easy/images/000032.png").convert("RGB")
    ground_truth_image = Image.open("/home/yuxin/CV_and_Sensing/final/Dataset_25/Easy/masks/000032.png").convert("L")
    blurred_image = image.copy().filter(ImageFilter.GaussianBlur(radius=2))
    hsv_img = blurred_image.convert("HSV")
    # binary_image = threshoding(image, threshold=[0, 120])
    # plt.imshow(binary_image, cmap='gray')
    # plt.axis('off')
    # plt.show()
    # region_growing(image, seed_point=(500, 500), threshold=50)

    # binary_seeds = get_safe_ocean_seeds(image)
    # plt.imshow(binary_seeds, cmap='gray')
    # plt.axis('off')
    # plt.show()

    # segmented_images = []
    # for y in range(0, binary_seeds.shape[0], 50):
    #     for x in range(0, binary_seeds.shape[1], 50):
    #         if binary_seeds[y, x] == 255:
    #             # print(f"Seed found at: ({x}, {y})")
    #             segmented_image = region_growing(image, seed_point=(x, y), threshold=30)
    #             segmented_images.append(segmented_image)
    # # Combine all segmented images
    # combined_segmented_image = np.zeros_like(binary_seeds)
    # for seg_img in segmented_images:
    #     combined_segmented_image = np.maximum(combined_segmented_image, seg_img)
    # plt.imshow(combined_segmented_image, cmap='gray')
    # plt.axis('off')
    # plt.show()

    # combined_segmented_image = threshold_and_region_growing(image, threshold=30)
    
    YoudensJ_evaluation(hsv_img, ground_truth_image)

    # draw_roc_curve(image, ground_truth_image)

    # mask = threshold_and_region_growing(hsv_img, threshold=60, saturation_threshold=190)
    # plt.imshow(mask, cmap='gray')
    # plt.axis('off')
    # plt.show()

if __name__ == "__main__":
    main()