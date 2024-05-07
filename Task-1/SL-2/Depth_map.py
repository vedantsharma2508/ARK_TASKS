import cv2
import os
from matplotlib import pyplot as plt
import numpy as np


# Load stereo image pair
left = cv2.imread('left_image.jpg', cv2.IMREAD_GRAYSCALE)
right = cv2.imread('right_image.jpg', cv2.IMREAD_GRAYSCALE)

img_height = 360
img_width = 640


def rectify_images(left, right, K1, D1, K2, D2, R, T):
    # Rectify stereo image pair
    image_size = left.shape[:2][::-1]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T)
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
    left_img_rectified = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
    right_img_rectified = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)
    return left_img_rectified, right_img_rectified



# Load camera intrinsic parameters and stereo calibration data
K1 = np.load('K1.npy')  # Intrinsic matrix of left camera
D1 = np.load('D1.npy')  # Distortion coefficients of left camera
K2 = np.load('K2.npy')  # Intrinsic matrix of right camera
D2 = np.load('D2.npy')  # Distortion coefficients of right camera
R = np.load('R.npy')    # Rotation matrix
T = np.load('T.npy')    # Translation vector

# Step 1: Image Rectification
left_img_rectified, right_img_rectified = rectify_images(left, right, K1, D1, K2, D2, R, T)

# Now continue with the rest of the steps (feature extraction, matching, disparity estimation, etc.

