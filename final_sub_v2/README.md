# Computer Vision and Sensing Final Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Download Dataset
Download the dataset from [here](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabh17_ucl_ac_uk/EXX-dnPoA11Bj1pP_rA_TmsB5YtD635w_84l3QaMwc-3Ig?e=VRUE66) and extract it to the project directory.

## Task 1
`task_1.py` contains the implementation for Task 1: Circle Detection using Colour Thresholding and Hough Transform.
- test_colour_thresholding(hsv_image, bgr_image, ground_mask, save_dir=save_dir)
- test_hough_circle_mask(bgr_image, gray_image, ground_mask, param1=50, param2=30, min_radius=300, max_radius=500, save_dir=save_dir)
- test_colour_thresholding_hough_circle(hsv_image, gray_image, bgr_image, ground_mask, save_dir=save_dir)

- roc_curve(image_dir, save_dir=save_dir, dataset_name='Medium')

- best_jouden_index, best_hue, best_param2 = YoudensJ_evaluation(image_dir="./Dataset_25/Easy/", save_dir=save_dir)

- test_img_path = "./Dataset_25/calibration_image_02_cam2_t1d.jpg"
    test_img_path = "./Dataset_25/test_image_2.jpg"

    bgr_test_image = cv2.imread(test_img_path)
    blurred_test_image = cv2.GaussianBlur(bgr_test_image, (5, 5), 0)
    hsv_test_image = cv2.cvtColor(blurred_test_image, cv2.COLOR_BGR2HSV)
    gray_test_image = cv2.cvtColor(blurred_test_image, cv2.COLOR_BGR2GRAY)

    test_1d_colour_thresholding_hough_circle(hsv_test_image, gray_test_image, bgr_test_image, save_dir=save_dir)

```bash
python task_1.py
```

# Task 2
`task_2.py` contains the implementation for Task 2: Measure the Projection Point of the Rotation Axis and the Height of the AO.
- video_to_frames(video_path, sample_rate=sample_rate)
- measure_swing(sample_rate=sample_rate, save_dir=save_dir)
- measure_distance()

```bash
python task_2.py
```