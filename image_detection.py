from ultralytics import YOLO
import os

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Make sure the correct model path is provided

def detect_objects_in_image(image_path):
    # Perform object detection
    results = model(image_path)

    # Get the filename and path to save the detected image
    image_filename = os.path.basename(image_path)
    detected_image_path = os.path.join('static/detections', image_filename)

    # Save the annotated image manually
    annotated_image = results[0].plot()  # Plot the image with bounding boxes
    results[0].save(detected_image_path)  # Save the annotated image

    return detected_image_path
