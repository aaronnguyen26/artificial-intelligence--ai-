import cv2
from ultralytics import YOLO
import time
import numpy as np # Import numpy for mask processing

# --- Configuration ---
# Path to your YOLOv8 model weights.
# For initial testing, this is 'models/yolov8n.pt' for general object detection.
# After training your custom food waste model, you'll update this to:
# 'runs/segment/food_waste_segmentation_v1/weights/best.pt'
MODEL_PATH = '/Users/minhnguyen/Desktop/Coding /artificial intelligence (ai)/ai_waste_detection/models/yolov8n.pt' 

# Set camera source:
# 0 for default webcam
# 1, 2, etc. for other connected cameras
# URL for IP camera (e.g., 'rtsp://user:password@ip_address:port/stream')
# Or path to a video file (e.g., 'video.mp4')
CAMERA_SOURCE = 0 

# Confidence threshold to filter detections (only show detections with confidence > this value)
CONF_THRESHOLD = 0.5 

# --- Initialize Model ---
try:
    # Load the YOLO model. If it's a segmentation model, it will automatically handle segmentation outputs.
    model = YOLO(MODEL_PATH) 
    print(f"YOLOv8 model loaded successfully: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading YOLOv8 model from {MODEL_PATH}: {e}")
    print("Please ensure the model file exists at the specified path.")
    exit()

# --- Initialize Camera ---
cap = cv2.VideoCapture(CAMERA_SOURCE)

# Adding a small delay can sometimes help with camera initialization on some systems
time.sleep(1) 

# Check if camera opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video source {CAMERA_SOURCE}.")
    print("Please check if the camera is connected and not in use by another application.")
    print("On macOS, ensure Terminal.app or VS Code.app (or Python) has camera permissions in System Settings -> Privacy & Security -> Camera.")
    print("Also, try changing CAMERA_SOURCE to 1 or 2.")
    exit()

# Optional: Set desired frame width and height for performance
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(f"Camera opened successfully. Starting real-time detection on source: {CAMERA_SOURCE}")
print("Press 'q' to quit the live feed.")

# --- Real-time Detection Loop ---
while True:
    ret, frame = cap.read() # Read a frame from the camera

    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Perform inference
    # 'stream=True' is crucial for efficient real-time processing, yielding results as they are available.
    # 'source=frame' tells the model to run inference on the current OpenCV frame.
    # 'save_txt=False' and 'save=False' prevent saving prediction files/images to disk unless you need them.
    results = model.predict(source=frame, show=False, conf=CONF_THRESHOLD, stream=True, save_txt=False, save=False)

    # Process results (iterate through results from a single frame)
    for r in results: # 'r' is a Results object for the current frame
        
        # Get bounding box coordinates, confidence scores, and class IDs
        boxes = r.boxes.xyxy.cpu().numpy()  # Bounding box coordinates (x1, y1, x2, y2)
        confidences = r.boxes.conf.cpu().numpy() # Confidence scores
        class_ids = r.boxes.cls.cpu().numpy()   # Class IDs

        # Check if segmentation masks are available (they will be after training your 'seg' model)
        masks = r.masks.xy if r.masks is not None else None # Pixel-level segmentation masks

        # Iterate through each detected object in the current frame
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            confidence = confidences[i]
            class_id = int(class_ids[i])
            
            # Get class name
            class_name = model.names[class_id] 
            
            # Define color for bounding box (you can customize this - BGR format)
            color = (0, 255, 0) # Green for bounding box

            # Draw bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Prepare label text (class name and confidence)
            label = f"{class_name}: {confidence:.2f}"
            
            # Put label text above the bounding box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- Segmentation Mask Visualization (Applies after 'seg' model is trained) ---
            if masks is not None and i < len(masks):
                # Get the polygon for the current object
                segment_polygon = masks[i] 
                
                # Convert the normalized polygon coordinates to absolute pixel coordinates
                # and reshape for cv2.fillPoly
                # Note: masks[i] usually gives you a list of [x, y] pairs for the polygon.
                # You might need to scale them by frame dimensions.
                # r.masks.xy is already scaled, but might need int conversion and reshaping
                
                # Example for drawing filled polygon (if r.masks.xy provides correct format):
                # Ensure the polygon points are integers and in the correct shape for fillPoly
                polygon_pts = np.array(segment_polygon, np.int32)
                
                # Reshape to (N, 1, 2) if necessary, for cv2.fillPoly to work with multiple contours
                polygon_pts = polygon_pts.reshape((-1, 1, 2))
                
                # Draw the filled polygon mask
                overlay = frame.copy()
                cv2.fillPoly(overlay, [polygon_pts], (255, 0, 0)) # Blue mask
                
                # Blend the overlay with the original frame to make it semi-transparent
                alpha = 0.3 # Transparency factor
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # --- Quantification Logic (Placeholder - Develop after training) ---
            # Once your model is trained for segmentation, you can use the 'masks'
            # to get the pixel area of each food waste item.
            # Example:
            if masks is not None and i < len(masks):
                # This gives the raw pixel coordinates of the segmentation mask polygon
                # To get area, you might need to convert it to a binary mask image
                # and then count non-zero pixels.
                
                # Example (conceptual, requires more precise mask handling):
                # mask_image = r.masks.data[i].cpu().numpy() # This is the binary mask (0/1)
                # pixel_area = np.sum(mask_image > 0)
                # print(f"  Pixel Area of {class_name}: {pixel_area}")
                # You'd then convert pixel_area to a more meaningful unit (e.g., grams)
                # using a calibration factor from your physical setup.
                pass # Placeholder for future quantification logic

    # Display the frame with detections
    cv2.imshow('Live Food Waste Detection', frame)

    # Wait for 1 millisecond. If 'q' key is pressed, break the loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release() # Release the camera resources
cv2.destroyAllWindows() # Close all OpenCV display windows
print("Live detection stopped. Resources released.")