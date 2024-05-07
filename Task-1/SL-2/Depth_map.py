import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_depth_map(left_image_path, right_image_path):
    # Load left and right images
    left_img = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(left_img, right_img)
    
    min_disp = disparity.min()
    max_disp = disparity.max()
    disparity_normalized = np.uint8((disparity - min_disp) * (255 / (max_disp - min_disp)))
    
    # Invert disparity map to represent depth
    depth_map = cv2.bitwise_not(disparity_normalized)
    
    return depth_map

def generate_heatmap(depth_map):
    # Apply colormap to depth map
    heatmap = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    
    return heatmap

def visualize_heatmap(heatmap):
    # Display the heatmap
    plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Save the heatmap image
    cv2.imwrite('heatmap_image.png', heatmap)

# Paths to left and right camera images
left_image_path = 'left.png'
right_image_path = 'right.png'

# Generate depth map
depth_map = generate_depth_map(left_image_path, right_image_path)

# Generate heatmap from depth map
heatmap = generate_heatmap(depth_map)

# Visualize the heatmap
visualize_heatmap(heatmap)

