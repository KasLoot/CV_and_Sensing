# Task 1: Image Segmentation and Circle Detection

This folder contains the Python script `task_1.py`, which is a conversion of the `task_1.ipynb` notebook. It performs various image segmentation and circle detection techniques on a sample image.

## Prerequisites

Ensure you have the following Python libraries installed:

- `torchvision`
- `Pillow` (PIL)
- `matplotlib`
- `numpy`
- `tqdm`
- `opencv-python` (cv2)

You can install them using pip:

```bash
pip install torchvision pillow matplotlib numpy tqdm opencv-python
```

## Usage

To run the script, execute the following command from this directory:

```bash
python task_1.py
```

## Description

The script performs the following operations:

1.  **Image Loading**: Loads a sample image and its ground truth mask from `./Dataset_25/Easy/`.
2.  **Colour Thresholding**:
    - Hue Channel only.
    - Hue + Saturation + Value (HSV).
3.  **Region Growing**: Segments the image using region growing from a seed point.
4.  **Hough Circle Detection**: Detects circles using the Hough Transform.
5.  **RANSAC Circle Detection**: Fits a circle using RANSAC algorithm.
6.  **Combined Methods**:
    - Colour Thresholding + Region Growing.
    - Thresholding + Region Growing + RANSAC.
7.  **Evaluation**:
    - Visualizes the overlap between prediction and ground truth.
    - Generates ROC curves for different methods.
    - Calculates Youden's J Index to find optimal parameters.

## Output

# Task 2: Centre Swing Detection and Depth Estimation

This folder contains the Python script `task_2.py`, which is a conversion of the `task_2.ipynb` notebook. It performs centre swing detection and depth estimation using stereo vision.

## Prerequisites

Ensure you have the following Python libraries installed:

- `torchvision`
- `Pillow` (PIL)
- `matplotlib`
- `numpy`
- `tqdm`
- `opencv-python` (cv2)
- `scipy`

You can install them using pip:

```bash
pip install torchvision pillow matplotlib numpy tqdm opencv-python scipy
```

## Usage

To run the script, execute the following command from this directory:

```bash
python task_2.py
```

## Description

The script is divided into two main functions:

1.  **`run_task_2b()`**:

    - **Centre Swing Detection**: Detects the centre of a swing in a sequence of images.
    - **Method**: Uses HSV thresholding, region growing, and RANSAC circle fitting to find the swing centre in each frame.
    - **Output**: Generates a plot `centre_swing.png` showing the trajectory of the swing centre.

2.  **`run_task_2c()`**:
    - **Depth Estimation**: Measures the distance to the ground using stereo images.
    - **Method**:
      - Applies lens distortion correction to the stereo images.
      - Computes the disparity map using Dynamic Programming.
      - Calculates the depth map from the disparity map.
      - Estimates the depth of a specific region of interest (ROI).
    - **Output**: Displays the disparity map and depth map, and prints the estimated depth.

By default, both tasks are executed when running the script. You can modify the `if __name__ == '__main__':` block in `task_2.py` to run only one of them.

The script will display several matplotlib figures showing the results of each step. It will also save a heatmap of Youden's J Index as `youdens_j_heatmap.png`.
