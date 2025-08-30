import cv2
import torch
from torch import load
from model import DETR
import albumentations as A
from utils.boxes import rescale_bboxes
from utils.setup import get_classes, get_colors
from utils.logger import get_logger
from utils.rich_handlers import DetectionHandler, create_detection_live_display
import sys
import time 


# Initialize logger and handlers
logger = get_logger("realtime")
detection_handler = DetectionHandler()

logger.print_banner()
logger.realtime("Initializing real-time sign language detection...")

transforms = A.Compose(
        [   
            A.Resize(224,224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.ToTensorV2()
        ]
    )

model = DETR(num_classes=3)
model.eval()
model.load_pretrained('pretrained/4426_model.pt')
CLASSES = get_classes() 
COLORS = get_colors() 

logger.realtime("Starting camera capture...")
cap = cv2.VideoCapture(0)

# Initialize performance tracking
frame_count = 0
fps_start_time = time.time()

while cap.isOpened(): 
    ret, frame = cap.read()
    if not ret:
        logger.error("Failed to read frame from camera")
        break
        
    # Time the inference
    inference_start = time.time()
    transformed = transforms(image=frame)
    result = model(torch.unsqueeze(transformed['image'], dim=0))
    inference_time = (time.time() - inference_start) * 1000  # Convert to ms

    probabilities = result['pred_logits'].softmax(-1)[:,:,:-1] 
    max_probs, max_classes = probabilities.max(-1)
    keep_mask = max_probs > 0.8

    batch_indices, query_indices = torch.where(keep_mask) 

    bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices,:], (1920,1080))
    classes = max_classes[batch_indices, query_indices]
    probas = max_probs[batch_indices, query_indices]

    # Prepare detection results for logging
    detections = []
    for bclass, bprob, bbox in zip(classes, probas, bboxes): 
        bclass_idx = bclass.detach().numpy()
        bprob_val = bprob.detach().numpy() 
        x1,y1,x2,y2 = bbox.detach().numpy()
        
        detections.append({
            'class': CLASSES[bclass_idx],
            'confidence': float(bprob_val),
            'bbox': [float(x1), float(y1), float(x2), float(y2)]
        })
        
        # Draw bounding boxes on frame
        frame = cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), COLORS[bclass_idx], 10)
        frame_text = f"{CLASSES[bclass_idx]} - {round(float(bprob_val),4)}"
        frame = cv2.rectangle(frame, (int(x1),int(y1)-100), (int(x1)+700,int(y1)), COLORS[bclass_idx], -1)
        frame = cv2.putText(frame, frame_text, (int(x1),int(y1)), cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,255), 4, cv2.LINE_AA)

    # Calculate FPS
    frame_count += 1
    if frame_count % 30 == 0:  # Log every 30 frames
        elapsed_time = time.time() - fps_start_time
        fps = 30 / elapsed_time
        
        # Log detection results and performance
        if detections:
            detection_handler.log_detections(detections, frame_id=frame_count)
        detection_handler.log_inference_time(inference_time, fps)
        
        # Reset FPS counter
        fps_start_time = time.time()

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        logger.realtime("Stopping real-time detection...")
        break

cap.release() 
cv2.destroyAllWindows() 
