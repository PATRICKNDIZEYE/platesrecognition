from ultralytics import YOLO
import cv2
import numpy as np

import util
from sort.sort import *
from util import get_car, write_csv

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
# license_plate_detector = YOLO('./models/license_plate_detector.pt')  # Commented out for testing

# load video
cap = cv2.VideoCapture('./sample.mp4')

vehicles = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # For testing, just save vehicle detections without license plate detection
        for track_id in track_ids:
            x1, y1, x2, y2, track_id = track_id
            results[frame_nmr][int(track_id)] = {'car': {'bbox': [x1, y1, x2, y2]}}

        # Comment out license plate detection for now
        # license_plates = license_plate_detector(frame)[0]
        # for license_plate in license_plates.boxes.data.tolist():
        #     x1, y1, x2, y2, score, class_id = license_plate
        #     # ... rest of license plate processing

# write results
write_csv(results, './test_vehicles_only.csv')
print("Vehicle detection and tracking completed! Check test_vehicles_only.csv for results.") 