from ultralytics import YOLO
import os

def train_model():
    model = YOLO(r'runs\detect\boulder_detection7\weights\last.pt')
    config_path = r'E:\AI_Bouldering_Overlay\Climbing Holds and Volumes.v1i.yolov11\data.yaml'  


    results = model.train(data=config_path,
                          epochs=100,
                          imgsz=640,
                          batch=16,
                          name='boulder_detection',
                          device='0',
                          augment=True,
                          workers=0, 
                          patience=30,
                          lr0=0.01,
                          cos_lr=True
                          )
    return results

if __name__ == '__main__':
    train_model()