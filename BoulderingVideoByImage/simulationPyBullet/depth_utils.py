import cv2
import numpy as np


def create_depth_map_from_wall(imagePath, wall_height_meters = 4.0):
    # only for testing replace this with real depth data from 3d scan

    img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    
    # Simulate depth - closer to top (higher y) = farther from camera
    depth_map = np.zeros((height, width))
    for y in range(height):
        depth_map[y,:] = wall_height_meters * (1 - y/height)  # Linear gradient
        
    return depth_map

def save_depth_map(depth_map, output_path):
    """Saves depth map as a 16-bit PNG for precision"""
    normalized = (depth_map * 65535 / depth_map.max()).astype(np.uint16)
    cv2.imwrite(output_path, normalized)