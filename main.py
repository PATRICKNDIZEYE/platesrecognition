from ultralytics import YOLO
import cv2
import time
import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import numpy as np

# Configuration
SKIP_FRAMES = 0  # Process every frame for better detection
DISPLAY_OUTPUT = True

results = {}
mot_tracker = Sort()

# Load models once at the start
print("Loading models...")
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# Load video
cap = cv2.VideoCapture('./sample.mp4')
original_fps = cap.get(cv2.CAP_PROP_FPS)
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Original video: {original_width}x{original_height} @ {original_fps:.2f} FPS")

# Calculate new dimensions while maintaining aspect ratio
target_width = 1280
scale_factor = target_width / original_width
new_width = int(original_width * scale_factor)
new_height = int(original_height * scale_factor)
print(f"Processing at: {new_width}x{new_height}")

vehicles = [2, 3, 5, 7]  # COCO class IDs for vehicles
frame_count = 0
frame_nmr = -1
start_time = time.time()

# Store last detected plates for smoother display
last_detected_plates = {}
plate_display_duration = 30  # frames to keep displaying a plate after detection

print("Starting video processing... (Press Q to quit)")
ret = True
while ret:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    frame_start_time = time.time()
    
    # Skip frames if needed (set SKIP_FRAMES > 0)
    if frame_count % (SKIP_FRAMES + 1) != 0:
        continue
        
    frame_nmr += 1
    results[frame_nmr] = {}
    
    # Resize frame for processing
    frame_resized = cv2.resize(frame, (new_width, new_height))
    
    # Make a copy for display to avoid modifying the processing frame
    display_frame = frame_resized.copy()
    
    # Detect vehicles with lower confidence threshold
    detections = coco_model(frame_resized, conf=0.3, verbose=False)[0]
    detections_ = []
    
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles and score > 0.3:
            detections_.append([x1, y1, x2, y2, score])
            # Draw vehicle boxes in blue
            if DISPLAY_OUTPUT:
                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # Track vehicles
    track_ids = mot_tracker.update(np.asarray(detections_))

    # Detect license plates with lower confidence threshold
    license_plates = license_plate_detector(
        frame_resized, 
        conf=0.1,  # Lower confidence threshold for better detection
        iou=0.3,   # Lower IoU for more detections
        verbose=False
    )[0]
    
    current_frame_plates = {}
    
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        
        # Draw all detected license plates in yellow
        if DISPLAY_OUTPUT and score > 0.1:
            cv2.rectangle(display_frame, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        (0, 255, 255),  # Yellow box
                        2)
        
        # Process high confidence detections for OCR
        if score > 0.3:
            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            
            if car_id != -1:
                # Store plate position for this car
                plate_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                current_frame_plates[car_id] = {
                    'bbox': [x1, y1, x2, y2],
                    'center': plate_center,
                    'score': score,
                    'car_bbox': [xcar1, ycar1, xcar2, ycar2]
                }
                
                # Process the plate for OCR
                pad = 5
                h, w = frame_resized.shape[:2]
                x1_pad = max(0, int(x1) - pad)
                y1_pad = max(0, int(y1) - pad)
                x2_pad = min(w, int(x2) + pad)
                y2_pad = min(h, int(y2) + pad)
                
                license_plate_crop = frame_resized[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if license_plate_crop.size > 0:
                    # Process license plate with better preprocessing
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    
                    # Apply adaptive thresholding for better text recognition
                    license_plate_crop_thresh = cv2.adaptiveThreshold(
                        license_plate_crop_gray, 255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        11, 2
                    )
                    
                    # Read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                    
                    if license_plate_text:
                        # Store the result
                        results[frame_nmr][car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': license_plate_text,
                                'bbox_score': score,
                                'text_score': license_plate_text_score
                            }
                        }
                        
                        # Update the current frame plates with text
                        current_frame_plates[car_id]['text'] = license_plate_text
                        current_frame_plates[car_id]['text_score'] = license_plate_text_score
                        
                        # Draw green box and text for successful reads
                        if DISPLAY_OUTPUT:
                            # Draw green box
                            cv2.rectangle(display_frame, 
                                        (int(x1), int(y1)), 
                                        (int(x2), int(y2)), 
                                        (0, 255, 0), 2)
                            
                            # Draw text with background for better visibility
                            text = f"{license_plate_text} ({score:.1f})"
                            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            cv2.rectangle(display_frame, 
                                        (int(x1), int(y1) - text_height - 10), 
                                        (int(x1) + text_width + 5, int(y1)), 
                                        (0, 0, 0), -1)  # Black background
                            cv2.putText(display_frame, 
                                      text, 
                                      (int(x1) + 2, int(y1) - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.7, (0, 255, 0), 2)  # Green text
    
    # Update last detected plates
    for car_id, plate_info in current_frame_plates.items():
        if 'text' in plate_info:  # Only keep plates with text
            last_detected_plates[car_id] = {
                'info': plate_info,
                'frames_since_detection': 0
            }
    
    # Draw plates from previous frames that are still relevant
    if DISPLAY_OUTPUT:
        plates_to_remove = []
        for car_id, data in last_detected_plates.items():
            data['frames_since_detection'] += 1
            if data['frames_since_detection'] > plate_display_duration:
                plates_to_remove.append(car_id)
                continue
                
            plate_info = data['info']
            x1, y1, x2, y2 = plate_info['bbox']
            
            # Fade out effect based on frames since detection
            alpha = max(0.3, 1.0 - (data['frames_since_detection'] / plate_display_duration) * 0.7)
            
            # Draw semi-transparent green box
            overlay = display_frame.copy()
            cv2.rectangle(overlay, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        (0, int(255 * alpha), 0), -1)  # Fading green fill
            cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
            
            # Draw border
            cv2.rectangle(display_frame, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        (0, 255, 0), 2)
            
            # Draw text if available
            if 'text' in plate_info:
                text = f"{plate_info['text']} ({plate_info['score']:.1f})"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                # Text background
                cv2.rectangle(display_frame, 
                            (int(x1), int(y1) - text_height - 10), 
                            (int(x1) + text_width + 5, int(y1)), 
                            (0, 0, 0), -1)
                
                # Text
                cv2.putText(display_frame, 
                          text, 
                          (int(x1) + 2, int(y1) - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 255, 0), 2)
        
        # Remove old plates
        for car_id in plates_to_remove:
            last_detected_plates.pop(car_id, None)
    
    # Display FPS and frame info
    if DISPLAY_OUTPUT:
        fps = 1.0 / (time.time() - frame_start_time) if (time.time() - frame_start_time) > 0 else 0
        
        # Draw info panel
        info_text = f"FPS: {fps:.1f} | Frame: {frame_nmr} | Plates: {len(license_plates)}"
        cv2.rectangle(display_frame, (5, 5), (400, 60), (0, 0, 0), -1)
        cv2.putText(display_frame, info_text, 
                   (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        # Add legend
        legend = "Blue: Vehicles | Yellow: Detected Plates | Green: Read Plates"
        cv2.putText(display_frame, legend, 
                   (10, display_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        cv2.imshow('ANPR - License Plate Recognition', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Calculate and print final performance metrics
processing_time = time.time() - start_time
print(f"\nProcessing complete!")
print(f"Total frames processed: {frame_nmr + 1}")
print(f"Total processing time: {processing_time:.2f} seconds")
print(f"Average FPS: {(frame_nmr + 1) / processing_time:.2f}")

# Release resources
cap.release()
cv2.destroyAllWindows()

# Write results
write_csv(results, './test.csv')
print("Results saved to test.csv")