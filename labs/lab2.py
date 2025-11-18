import numpy as np
import copy
import cv2
from matplotlib import pyplot as plt
import random
from PIL import Image
#to visualize the plots within the notebook
# for making plots looking nicer
plt.style.use('fivethirtyeight')



def draw_corners(image, corners_map, color=(255, 0, 0)):
    """Draw a point for each possible corner.
    color image: [H, W, 3]
    corners_map: list of pixel coordinates (xy indexing)
    """
    
    color_img = cv2.cvtColor(image[:,:,0], cv2.COLOR_GRAY2BGR)
    for corner in corners_map:
        cv2.circle(color_img, corner, 1, color, -1)
    return color_img

"""
Utility functions here, you can use code from the last week. FFT implementation is reccomended as it is faster.
"""
def get_mean_filter_kernel(h, w):
    kernel = np.ones((h, w), dtype=np.float32) / (h * w)
    return kernel


def get_gaussian_kernel(h, w, sigma):
    assert (h % 2 == 1) and (w % 2 == 1), print("kernal size must be odd number!")
    kernel = np.zeros((h, w), dtype=np.float32)

    m = h // 2
    n = w // 2
    for i in range(-m, m + 1):
        for j in range(-n, n + 1):
            kernel[i + m, j + n] = np.exp(-(i**2 + j**2) / (2 * sigma**2))/(2 * np.pi * sigma**2)
    kernel /= np.sum(kernel)
    return kernel

"""
Implement your moravec detector here.
"""
import numpy as np

def moravec(image, window_size=3, threshold=100., weights=None):
    """Moravec's corner detection for each pixel of the image.
    image: grayscale image: [H, W]
    """
    assert window_size % 2 == 1, "window size should be an odd number"
    H, W = image.shape
    corners = np.zeros((H, W), dtype=np.float32)
    half_w = window_size // 2
    
    # Pad with extra margin for shifts
    padded_img = np.pad(image, half_w + 1, mode='constant', constant_values=0)
    
    # Process each pixel in the original image
    for i in range(half_w + 1, padded_img.shape[0] - half_w - 1):
        for j in range(half_w + 1, padded_img.shape[1] - half_w - 1):
            
            # Extract patch centered at current position
            patch = padded_img[i-half_w:i+half_w+1, j-half_w:j+half_w+1]
            if weights is not None:
                patch = patch * weights
            
            min_ssd = float('inf')
            
            # Check all 8 directions
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), 
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                shifted_patch = padded_img[i+dx-half_w:i+dx+half_w+1, 
                                          j+dy-half_w:j+dy+half_w+1]
                if weights is not None:
                    shifted_patch = shifted_patch * weights
                    
                ssd = np.sum((patch - shifted_patch) ** 2)
                min_ssd = min(min_ssd, ssd)
            
            # Map back to original image coordinates
            orig_i = i - half_w - 1
            orig_j = j - half_w - 1
            corners[orig_i, orig_j] = min_ssd
            
            # Apply threshold
            if corners[orig_i, orig_j] < threshold:
                corners[orig_i, orig_j] = 0
    
    # Extract corner points
    corner_points = []
    for i in range(H):
        for j in range(W):
            if corners[i, j] != 0:
                corner_points.append((j, i))  # xy indexing
    
    return corner_points
    

img = cv2.imread('../dataset/building.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

moravec_corners = moravec(gray_img, threshold=500.)
img_with_corner = draw_corners(img, moravec_corners)

moravec_corners2 = moravec(gray_img, threshold=20., weights=get_gaussian_kernel(3, 3, 1.0))
img_with_corner2 = draw_corners(img, moravec_corners2)

# Show Original and detected corners on image
fig, axis = plt.subplots(1, 3, figsize=(20, 8), sharey=False)
axis[0].imshow(gray_img, 'gray')
axis[0].grid(False)
axis[0].set_title('Gray scale image')
axis[0].set_axis_off()

axis[1].imshow(img_with_corner, 'gray')
axis[1].grid(False)
axis[1].set_title('Moravec corner detection (uniform)')
axis[1].set_axis_off()

axis[2].imshow(img_with_corner2, 'gray')
axis[2].grid(False)
axis[2].set_title('Moravec corner detection (gaussian)')
axis[2].set_axis_off()

plt.show()