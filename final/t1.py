import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import cv2

import torch
import torchvision.transforms as transforms
from collections import deque



def thresholding():

    image = Image.open("/home/yuxin/CV_and_Sensing/final/Dataset_25/Easy/images/000016.png").convert('L')
    image = np.array(image).astype(np.float32)
    print("Image shape:", image.shape)

    blur = transforms.GaussianBlur(kernel_size=5, sigma=(1.0, 1.0))
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    smoothed_tensor = blur(image_tensor)
    smoothed_image = smoothed_tensor.squeeze(0).squeeze(0).numpy()

    hist, bins = np.histogram(smoothed_image, bins=256, range=(0, 256))

    threshold_1_low, threshold_1_up = 150, 185
    threshold_2_low, threshold_2_up = 55, 105
    binary_image = np.zeros(smoothed_image.shape, dtype=np.uint8)
    binary_image[((smoothed_image >= threshold_1_low) & (smoothed_image <= threshold_1_up)) |
                 ((smoothed_image >= threshold_2_low) & (smoothed_image <= threshold_2_up))
                 ] = 1
    # binary_image[((smoothed_image >= threshold_1_low) & (smoothed_image <= threshold_1_up))] = 1

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes[0].imshow(smoothed_image, cmap='gray')
    axes[0].set_title('Smoothed Grayscale Image')
    axes[0].axis('off')

    axes[1].plot(bins[:-1], hist, color='black')
    axes[1].set_title('Histogram of Smoothed Image')
    axes[1].set_xlabel('Pixel Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlim([0, 255])

    axes[2].imshow(binary_image, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Binary Mask')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def edge_detection(visualize=False):
    image = cv2.imread('/home/yuxin/CV_and_Sensing/final/Dataset_25/Easy/images/000016.png', cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (7, 7), sigmaX=1.0, sigmaY=1.0)

    # Canny Edge Detection
    edges = cv2.Canny(blurred_image, threshold1=0, threshold2=200)

    if visualize:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title("Original Grayscale Image")
        plt.imshow(image, cmap='gray')
        plt.subplot(1, 3, 2)
        plt.title("Canny Edge Detection")
        plt.imshow(edges, cmap='gray')

        # connect edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        plt.subplot(1, 3, 3)
        plt.title("Connected Edges")
        plt.imshow(edges, cmap='gray')
        plt.show()
    else:
        # connect edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    return edges


def region_growing_multi_seed(image, seed, threshold):
    masks = []
    for i, s in enumerate(seed):
        masks.append(region_growing(image, s, threshold[i]))
    
    # Combine all masks
    combined_mask = np.zeros_like(masks[0])
    for m in masks:
        combined_mask = cv2.bitwise_or(combined_mask, m)

    # plt.subplot(1, 2, 1)
    # plt.title("Original Image")
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.subplot(1, 2, 2)
    # plt.title("Combined Region Growing Mask")
    # plt.imshow(combined_mask, cmap='gray')
    # plt.show()

    return combined_mask


def region_growing(image, seed, threshold, visualize=False):
    h, w = image.shape[:2]
    visited = np.zeros((h, w), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Use deque for efficient FIFO operations
    queue = deque([seed])
    
    # Mark seed as visited and part of the region
    visited[seed[1], seed[0]] = 1
    mask[seed[1], seed[0]] = 255
    
    # Reference value from the seed
    seed_value = image[seed[1], seed[0]].astype(np.float32)
    
    # 8-connected neighbors
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]
    
    while queue:
        x, y = queue.popleft()
        
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < w and 0 <= ny < h:
                if not visited[ny, nx]:
                    # Calculate similarity
                    curr_val = image[ny, nx].astype(np.float32)
                    # Euclidean distance in RGB space
                    dist = np.linalg.norm(curr_val - seed_value)
                    
                    if dist < threshold:
                        visited[ny, nx] = 1
                        mask[ny, nx] = 255
                        queue.append((nx, ny))
                    else:
                        visited[ny, nx] = 1
                        
    # Display the result
    if visualize:
        plt.figure(figsize=(10,5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.subplot(1, 2, 2)
        plt.title("Region Growing Mask")
        plt.imshow(mask, cmap='gray')
        plt.show()
    
    return mask


def run_thresholding(image, threshold):
    # Simple thresholding on smoothed image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    return mask

def run_region_growing_only(image, threshold_multiplier):
    seed = [(500, 500), (570, 200), (509, 643), (787, 603), (529, 679)]
    threshold = [90.0, 65.0, 50.0, 50.0, 70.0]
    threshold = [t * threshold_multiplier for t in threshold]
    return region_growing_multi_seed(image, seed, threshold)

def run_edge_plus_region(image, threshold_multiplier):
    edges = edge_detection(visualize=False)

    seed = [(500, 500), (570, 200), (509, 643), (787, 603), (529, 679)]
    threshold = [90.0, 65.0, 50.0, 50.0, 70.0]
    threshold = [t * threshold_multiplier for t in threshold]
    region_mask_1 = region_growing_multi_seed(image, seed, threshold)

    seed = [(500, 500), (741, 123), (570, 200), (509, 643), (787, 603), (396, 363)]
    threshold = [115.0, 80.0, 80.0, 60.0, 50.0, 30.0]
    threshold = [t * threshold_multiplier for t in threshold]
    
    combined_image = image.copy()
    combined_image[edges > 0] = [0, 0, 0]
    region_mask_2 = region_growing_multi_seed(combined_image, seed, threshold)

    combined_mask = cv2.bitwise_or(region_mask_1, region_mask_2)
    return combined_mask

def calculate_roc_points(ground_truth, predict_func, param_range):
    tprs = []
    fprs = []
    best_dist = float('inf')
    best_mask = None
    
    gt_flat = ground_truth.flatten() / 255
    
    for param in param_range:
        predicted_mask = predict_func(param)
        pred_flat = predicted_mask.flatten() / 255
        
        tp = np.sum((gt_flat == 1) & (pred_flat == 1))
        fp = np.sum((gt_flat == 0) & (pred_flat == 1))
        tn = np.sum((gt_flat == 0) & (pred_flat == 0))
        fn = np.sum((gt_flat == 1) & (pred_flat == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tprs.append(tpr)
        fprs.append(fpr)
        
        # Check if this is the best mask (closest to (0,1))
        dist = np.sqrt((1 - tpr)**2 + fpr**2)
        if dist < best_dist:
            best_dist = dist
            best_mask = predicted_mask
        
    # Sort by FPR
    sorted_indices = np.argsort(fprs)
    fprs = np.array(fprs)[sorted_indices]
    tprs = np.array(tprs)[sorted_indices]
    
    # Add (0,0) and (1,1)
    fprs = np.concatenate(([0], fprs, [1]))
    tprs = np.concatenate(([0], tprs, [1]))
    
    return fprs, tprs, best_mask

def evaluate_roc():
    image_path = '/home/yuxin/CV_and_Sensing/final/Dataset_25/Easy/images/000016.png'
    ground_truth_path = '/home/yuxin/CV_and_Sensing/final/Dataset_25/Easy/masks/000016.png'
    
    image = cv2.imread(image_path)
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    _, ground_truth = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)
    
    plt.figure(figsize=(10, 8))
    
    # 1. Thresholding
    print("Evaluating Thresholding...")
    thresholds = np.linspace(0, 255, 20)
    fprs, tprs, best_mask_thresh = calculate_roc_points(ground_truth, lambda p: run_thresholding(image, p), thresholds)
    roc_auc = np.trapz(tprs, fprs)
    plt.plot(fprs, tprs, label=f'Thresholding (AUC = {roc_auc:.2f})')
    cv2.imwrite('best_mask_thresholding.png', best_mask_thresh)
    
    # 2. Region Growing
    print("Evaluating Region Growing...")
    multipliers = np.linspace(0.1, 3.0, 10)
    fprs, tprs, best_mask_rg = calculate_roc_points(ground_truth, lambda p: run_region_growing_only(image, p), multipliers)
    roc_auc = np.trapz(tprs, fprs)
    plt.plot(fprs, tprs, label=f'Region Growing (AUC = {roc_auc:.2f})')
    cv2.imwrite('best_mask_region_growing.png', best_mask_rg)
    
    # 3. Edge + Region Growing
    print("Evaluating Edge + Region Growing...")
    fprs, tprs, best_mask_edge_rg = calculate_roc_points(ground_truth, lambda p: run_edge_plus_region(image, p), multipliers)
    roc_auc = np.trapz(tprs, fprs)
    plt.plot(fprs, tprs, label=f'Edge + Region Growing (AUC = {roc_auc:.2f})')
    cv2.imwrite('best_mask_edge_region_growing.png', best_mask_edge_rg)
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.savefig('roc_comparison.png')
    plt.show()

def combine_edge_and_region():
    image = cv2.imread('/home/yuxin/CV_and_Sensing/final/Dataset_25/Easy/images/000016.png')
    edges = edge_detection()

    seed = [(500, 500), (570, 200), (509, 643), (787, 603), (529, 679)]  # Example seed point (x, y)
    threshold = [90.0, 65.0, 50.0, 50.0, 70.0]  # Similarity threshold
    region_mask_1 = region_growing_multi_seed(image, seed, threshold)

    seed = [(500, 500), (741, 123), (570, 200), (509, 643), (787, 603), (396, 363)]  # Example seed point (x, y)
    threshold = [115.0, 80.0, 80.0, 60.0, 50.0, 30.0]  # Similarity threshold
    # set edge pixels to black in the original image
    combined_image = image.copy()
    combined_image[edges > 0] = [0, 0, 0]
    region_mask_2 = region_growing_multi_seed(combined_image, seed, threshold)

    combined_mask = cv2.bitwise_or(region_mask_1, region_mask_2)
    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 4, 2)
    plt.title("Region Growing Mask")
    plt.imshow(region_mask_1, cmap='gray')
    plt.subplot(1, 4, 3)
    plt.title("Region Growing on Edge-Removed Image")
    plt.imshow(region_mask_2, cmap='gray')
    plt.subplot(1, 4, 4)
    plt.title("Combined Mask")
    plt.imshow(combined_mask, cmap='gray')
    plt.show()

    



if __name__ == "__main__":
    evaluate_roc()