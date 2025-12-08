import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

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

def colour_thresholding(hsv_image_array, lower_bound, upper_bound):
    """Apply colour thresholding to an HSV image."""
    mask = cv2.inRange(hsv_image_array, lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels < 2:
        return np.zeros_like(mask)

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    mask = (labels == largest_label).astype(np.uint8) * 255
    return mask

def colour_thresholding_batch(batch_array, lower_bound, upper_bound):
    """
    vectorized version of inRange for 4D arrays (N, H, W, C).
    """
    # 1. Create a boolean mask where pixels are >= lower bound
    # checks across the channel axis (axis 3)
    # result shape: (N, H, W, C) -> boolean
    lower_mask = np.all(batch_array >= lower_bound, axis=-1)
    
    # 2. Create a boolean mask where pixels are <= upper bound
    upper_mask = np.all(batch_array <= upper_bound, axis=-1)
    
    # 3. Combine both (Logical AND)
    final_mask = np.logical_and(lower_mask, upper_mask)
    
    # Convert to uint8 for OpenCV operations
    final_mask_uint8 = final_mask.astype(np.uint8) * 255
    
    processed_masks = []
    kernel = np.ones((5, 5), np.uint8)

    for i in range(final_mask_uint8.shape[0]):
        mask = final_mask_uint8[i]
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels < 2:
            processed_masks.append(np.zeros_like(mask))
            continue

        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        mask = (labels == largest_label).astype(np.uint8) * 255
        processed_masks.append(mask)
    
    return np.array(processed_masks)


def test_colour_thresholding(hsv_image, bgr_image, ground_mask):
    colour_thresholding_mask = colour_thresholding(
        hsv_image,
        lower_bound=np.array([55, 15, 40]),
        upper_bound=np.array([150, 255, 210])
    )
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Colour Thresholding Mask')
    plt.imshow(colour_thresholding_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    colour_thresholding_difference = cv2.absdiff(colour_thresholding_mask, ground_mask)
    plt.title('Difference with Ground Truth')
    plt.imshow(colour_thresholding_difference, cmap='gray')
    plt.axis('off')
    plt.show()

def test_colour_thresholding_batch(hsv_images, bgr_images, ground_masks):
    hsv_images_array = np.array(hsv_images)
    colour_thresholding_masks = colour_thresholding_batch(
        hsv_images_array,
        lower_bound=np.array([55, 15, 40]),
        upper_bound=np.array([150, 255, 210])
    )

    sample_index = 0  # Change this index to visualize different samples
    bgr_image = bgr_images[sample_index]
    colour_thresholding_mask = colour_thresholding_masks[sample_index]
    ground_mask = ground_masks[sample_index]
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title('Colour Thresholding Mask')
    plt.imshow(colour_thresholding_mask, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    colour_thresholding_difference = cv2.absdiff(colour_thresholding_mask, ground_mask)
    plt.title('Difference with Ground Truth')
    plt.imshow(colour_thresholding_difference, cmap='gray')
    plt.axis('off')
    plt.show()


def hough_circle_mask(gray_image, param2=30, min_radius=300, max_radius=500):

    h, w = gray_image.shape

    circles = cv2.HoughCircles(
        gray_image,
        cv2.HOUGH_GRADIENT,
        dp=2,
        minDist=h,
        param1=50,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    # Create empty mask
    mask = np.zeros_like(gray_image)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        x, y, r = circles[0]

        # Draw mask
        cv2.circle(mask, (x, y), r, 255, -1)

        # Bounding box
        x1 = max(x - r, 0)
        y1 = max(y - r, 0)
        x2 = min(x + r, gray_image.shape[1])
        y2 = min(y + r, gray_image.shape[0])

    else:
        # No circle found → return full white?
        mask[:] = 0
        x1, y1, x2, y2 = 0, 0, gray_image.shape[1], gray_image.shape[0]

    return mask, (x1, y1, x2, y2)


def test_hough_circle_mask(bgr_image, gray_image, ground_mask):
    hough_mask, box = hough_circle_mask(gray_image, min_radius=300, max_radius=500)

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Hough Circle Mask')
    plt.imshow(hough_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    hough_difference = cv2.absdiff(hough_mask, ground_mask)
    plt.title('Difference with Ground Truth')
    plt.imshow(hough_difference, cmap='gray')
    plt.axis('off')
    plt.show()


def colour_thresholding_hough_circle(hsv_image, gray_image, lower_bound, upper_bound, param2=30, min_radius=300, max_radius=500):
    """Combine colour thresholding and Hough Circle Masking."""
    colour_mask = colour_thresholding(hsv_image, lower_bound, upper_bound)

    hough_mask, _ = hough_circle_mask(gray_image, param2, min_radius, max_radius)
    combined_mask = cv2.bitwise_or(colour_mask, hough_mask)

    return combined_mask

def test_colour_thresholding_hough_circle(hsv_image, gray_image, bgr_image, ground_mask):
    combined_mask = colour_thresholding_hough_circle(
        hsv_image,
        gray_image,
        lower_bound=np.array([55, 15, 40]),
        upper_bound=np.array([150, 255, 210]),
        min_radius=300,
        max_radius=500
    )

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Combined Mask')
    plt.imshow(combined_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    combined_difference = cv2.absdiff(combined_mask, ground_mask)
    plt.title('Difference with Ground Truth')
    plt.imshow(combined_difference, cmap='gray')
    plt.axis('off')
    plt.show()


def roc_curve(images_dir):
    colour_thresholding_hue_thresholds = range(0, 256, 5)
    tp_counts_colour_thresholding = np.zeros(len(colour_thresholding_hue_thresholds))
    tn_counts_colour_thresholding = np.zeros(len(colour_thresholding_hue_thresholds))
    fp_counts_colour_thresholding = np.zeros(len(colour_thresholding_hue_thresholds))
    fn_counts_colour_thresholding = np.zeros(len(colour_thresholding_hue_thresholds))

    hough_circle_param2_thresholds = range(100, 10, -2)
    tp_counts_hough_circle = np.zeros(len(hough_circle_param2_thresholds))
    tn_counts_hough_circle = np.zeros(len(hough_circle_param2_thresholds))
    fp_counts_hough_circle = np.zeros(len(hough_circle_param2_thresholds))
    fn_counts_hough_circle = np.zeros(len(hough_circle_param2_thresholds))

    
    images_path = os.path.join(images_dir, 'images')
    masks_path = os.path.join(images_dir, 'masks')    
    image_files = sorted(os.listdir(images_path))
    
    for image_name in tqdm(image_files, desc="Processing images for ROC curve"):
        if not image_name.endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(images_path, image_name)
        mask_path = os.path.join(masks_path, image_name)
        
        bgr_image = cv2.imread(img_path)
        if bgr_image is None:
            continue
            
        ground_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if ground_mask is None:
            continue
            
        blurred = cv2.GaussianBlur(bgr_image, (5, 5), 0)
        hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        gray_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        upper_bound = np.array([255, 255, 255])
        
        for i, v in enumerate(colour_thresholding_hue_thresholds):
            lower_bound = np.array([v, 0, 0])
            
            colour_thresholding_mask = colour_thresholding(
                hsv_image,
                lower_bound=lower_bound,
                upper_bound=upper_bound
            )
            
            tp = np.sum((colour_thresholding_mask == 255) & (ground_mask == 255))
            tn = np.sum((colour_thresholding_mask == 0) & (ground_mask == 0))
            fp = np.sum((colour_thresholding_mask == 255) & (ground_mask == 0))
            fn = np.sum((colour_thresholding_mask == 0) & (ground_mask == 255))
            
            tp_counts_colour_thresholding[i] += tp
            tn_counts_colour_thresholding[i] += tn
            fp_counts_colour_thresholding[i] += fp
            fn_counts_colour_thresholding[i] += fn

        for i, param2 in enumerate(hough_circle_param2_thresholds):
            hough_mask, _ = hough_circle_mask(gray_image, param2=param2, min_radius=300, max_radius=500)

            tp = np.sum((hough_mask == 255) & (ground_mask == 255))
            tn = np.sum((hough_mask == 0) & (ground_mask == 0))
            fp = np.sum((hough_mask == 255) & (ground_mask == 0))
            fn = np.sum((hough_mask == 0) & (ground_mask == 255))
            
            tp_counts_hough_circle[i] += tp
            tn_counts_hough_circle[i] += tn
            fp_counts_hough_circle[i] += fp
            fn_counts_hough_circle[i] += fn


    tp_rate_colour_thresholding = tp_counts_colour_thresholding / (tp_counts_colour_thresholding + fn_counts_colour_thresholding + 1e-6)
    fp_rate_colour_thresholding = fp_counts_colour_thresholding / (fp_counts_colour_thresholding + tn_counts_colour_thresholding + 1e-6)

    plt.figure()
    plt.plot(fp_rate_colour_thresholding, tp_rate_colour_thresholding, label='Colour Thresholding ROC', color='blue')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Colour Thresholding ROC Curve')
    plt.legend()
    plt.savefig('colour_thresholding_roc_curve.png')
    print("Saved colour_thresholding_roc_curve.png")
    plt.show()

    tp_rate_hough_circle = tp_counts_hough_circle / (tp_counts_hough_circle + fn_counts_hough_circle + 1e-6)
    fp_rate_hough_circle = fp_counts_hough_circle / (fp_counts_hough_circle + tn_counts_hough_circle + 1e-6)
    plt.figure()
    plt.plot(fp_rate_hough_circle, tp_rate_hough_circle, label='Hough Circle ROC', color='orange')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Hough Circle ROC Curve')
    plt.legend()
    plt.savefig('hough_circle_roc_curve.png')
    print("Saved hough_circle_roc_curve.png")
    plt.show()

    # Combine both ROC curves
    plt.figure()
    plt.plot(fp_rate_colour_thresholding, tp_rate_colour_thresholding, label='Colour Thresholding ROC', color='blue')
    plt.plot(fp_rate_hough_circle, tp_rate_hough_circle, label='Hough Circle ROC', color='orange')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.savefig('roc_curve_comparison.png')
    print("Saved roc_curve_comparison.png")
    plt.show()


def YoudensJ_evaluation(hsv_image, gray_image, ground_truth_image):
    gt_array = np.array(ground_truth_image)

    colour_thresholding_hue_thresholds = range(0, 256, 5)
    upper_bound = np.array([255, 255, 255])
    hough_circle_param2_thresholds = range(100, 10, -2)
    
    heatmap_data = np.zeros((len(colour_thresholding_hue_thresholds), len(hough_circle_param2_thresholds)))

    best_jouden_index = -1
    best_saturation = -1
    best_region_growing_threshold = -1

    
    
    for i, hue in tqdm(enumerate(colour_thresholding_hue_thresholds), desc="Evaluating Youden's J Index", total=len(colour_thresholding_hue_thresholds)):
        for j, param2 in enumerate(hough_circle_param2_thresholds):
            lower_bound = np.array([hue, 0, 0])
            binary_image = colour_thresholding_hough_circle(hsv_image, gray_image, lower_bound, upper_bound, param2=param2, min_radius=300, max_radius=500)
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
                best_hue = hue
                best_param2 = param2
    
    print(f"Best Youden's J Index: {best_jouden_index} at Hue: {best_hue}, Param2: {best_param2}")

    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_data, extent=[100, 10, 255, 0], aspect='auto', cmap='viridis')
    plt.colorbar(label="Youden's J Index")
    plt.xlabel('Param2')
    plt.ylabel('Hue')
    plt.title("Youden's J Index Heatmap (Colour Thresholding + Hough Circle)")
    plt.savefig('youdens_j_heatmap.png')
    print("Heatmap saved as 'youdens_j_heatmap.png'")
    plt.show()

    return best_jouden_index, best_hue, best_param2

    


def main():

    image_dir = './Dataset_25/Easy/'
    sample_image_name = '000016.png'
    bgr_image = cv2.imread(f'{image_dir}/images/{sample_image_name}')
    blurred = cv2.GaussianBlur(bgr_image, (5, 5), 0)
    hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    ground_mask = cv2.imread(f'{image_dir}/masks/{sample_image_name}', cv2.IMREAD_GRAYSCALE)


    bgr_images_all = []
    hsv_images_all = []
    ground_masks_all = []
    images_path = os.path.join(image_dir, 'images')
    masks_path = os.path.join(image_dir, 'masks')
    image_files = sorted(os.listdir(images_path))
    for image_name in tqdm(image_files, desc="Loading all images"):
        if not image_name.endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(images_path, image_name)
        mask_path = os.path.join(masks_path, image_name)
        
        bgr_image = cv2.imread(img_path)
        if bgr_image is None:
            continue
            
        ground_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if ground_mask is None:
            continue
            
        blurred = cv2.GaussianBlur(bgr_image, (5, 5), 0)
        hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        bgr_images_all.append(bgr_image)
        hsv_images_all.append(hsv_image)
        ground_masks_all.append(ground_mask)

    test_colour_thresholding_batch(hsv_images_all, bgr_images_all, ground_masks_all)
    # test_hough_circle_mask(bgr_image, gray_image, ground_mask)
    # test_colour_thresholding_hough_circle(hsv_image, gray_image, bgr_image, ground_mask)

    # roc_curve(image_dir)

    # best_jouden_index, best_hue, best_param2 = YoudensJ_evaluation(hsv_image, gray_image, ground_mask)



if __name__ == '__main__':
    main()