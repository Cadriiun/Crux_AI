import torch
from ultralytics import YOLO
import cv2


print("PyTorch Version:", torch.__version__)
print("Is CUDA available?:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Check your PyTorch installation.")

# model of yolo
model = YOLO("yolo11n.pt")
# pose model for yolo  
pose_model = YOLO("yolo11n-pose.pt").to('cuda')

input_video = "input_video.mp4"
output_video = "output.mp4"

cap = cv2.VideoCapture("input_video.mp4")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter("output.mp4",fourcc,fps,(width,height))

current_frame = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No more frame Exiting")
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model(frame_rgb,verbose = False)


    annotated_frame = results[0].plot()
    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    out.write(annotated_frame_bgr)

    current_frame += 1
    print(f"Frame {current_frame}/{total_frames} \n")

cap.release()
out.release()
print(f"Done! Processed {current_frame} frames")