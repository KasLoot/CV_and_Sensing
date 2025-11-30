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


def edge_detection():
    image = cv2.imread('/home/yuxin/CV_and_Sensing/final/Dataset_25/Easy/images/000016.png', cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (7, 7), sigmaX=1.0, sigmaY=1.0)

    # Canny Edge Detection
    edges = cv2.Canny(blurred_image, threshold1=0, threshold2=200)

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


def region_growing(image, seed, threshold):
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
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.title("Region Growing Mask")
    plt.imshow(mask, cmap='gray')
    plt.show()

    return mask

def combine_edge_and_region():
    image = cv2.imread('/home/yuxin/CV_and_Sensing/final/Dataset_25/Easy/images/000016.png')
    edges = edge_detection()
    region_mask = region_growing_multi_seed(image)

    # set edge pixels to black in the original image
    combined_image = image.copy()
    combined_image[edges > 0] = [0, 0, 0]
    region_mask_2 = region_growing_multi_seed(combined_image)

    combined_mask = cv2.bitwise_or(region_mask, region_mask_2)

    plt.subplot(1, 6, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 6, 2)
    plt.title("Edge Detection")
    plt.imshow(edges, cmap='gray')
    plt.subplot(1, 6, 3)
    plt.title("Combined Removed Edges Image")
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 6, 4)
    plt.title("Region Growing Mask")
    plt.imshow(region_mask, cmap='gray')
    plt.subplot(1, 6, 5)
    plt.title("Region Growing on Edge-Removed Image")
    plt.imshow(region_mask_2, cmap='gray')
    plt.subplot(1, 6, 6)
    plt.title("Combined Mask")
    plt.imshow(combined_mask, cmap='gray')
    plt.show()

def combine_edge_and_region_2():
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
    combine_edge_and_region_2()