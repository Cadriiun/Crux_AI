import cv2
import numpy as np
from ultralytics import YOLO

class PoseProcessor:
    def __init__(self, model_path='yolo11n-pose.pt'):
        self.model = YOLO(model_path)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def process_frame(self, image):
        results = self.model(image)
        if len(results[0].keypoints) == 0:
            return None
        
        keypoints = results[0].keypoints.xy.cpu().numpy()[0]
        confidences = results[0].keypoints.conf.cpu().numpy()[0]
        
        # Normalize to [-1, 1] range
        h, w = image.shape[:2]
        normalized = np.zeros((17, 3))
        normalized[:, 0] = (keypoints[:, 0] - w/2) / (w/2)  # X
        normalized[:, 1] = (keypoints[:, 1] - h/2) / (h/2)  # Y 
        normalized[:, 2] = confidences                      # Confidence
        
        return {name: norm for name, norm in zip(self.keypoint_names, normalized)}