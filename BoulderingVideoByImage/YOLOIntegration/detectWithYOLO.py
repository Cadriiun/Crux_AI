from ultralytics import YOLO
import cv2
import numpy as np
'''
    This will detect holds using YOLO and filter by Color
'''


def filter_holds_by_color(image_path, target_color_rgb, threshold=30):
    model = YOLO(r"runs\detect\boulder_detection7\weights\best.pt")
    img = cv2.imread(image_path)
    results = model(img)
    
    holds = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        roi = img[y1:y2, x1:x2]
        mean_color = np.mean(roi, axis=(0, 1))[::-1]
        
        if np.linalg.norm(mean_color - target_color_rgb) < threshold:
            holds.append([(x1 + x2)/2, (y1 + y2)/2]) 
    
    return holds

if __name__ == '__main__':
    filter_holds_by_color("boulderingwall.jpg",(255,0,0))