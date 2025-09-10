from ultralytics import YOLO

model = YOLO(r'runs\detect\boulder_detection7\weights\best.pt')

results = model('boulderingwall.jpg')

results[0].show()
results[0].save('detectedBoulderingWall.jpg')