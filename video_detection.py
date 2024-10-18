
import os
import cv2
from ultralytics import YOLO

# YOLOv8 model load
model = YOLO('yolov8n.pt')

def detect_objects_in_video(video_path):
    # Perform detection on the video
    results = model(video_path)

    # Open the original video to extract frame details (width, height, fps)
    video_capture = cv2.VideoCapture(video_path)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Get the base name of the video file
    detection_video_path = os.path.join('static/detections', os.path.basename(video_path))

    # Create a VideoWriter to save the output video with bounding boxes
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for saving video
    out = cv2.VideoWriter(detection_video_path, fourcc, fps, (width, height))

    # Loop over each frame result in the video
    for result in results:
        # Convert the frame to an image with bounding boxes (as a NumPy array)
        frame_with_boxes = result.plot()  # Bounding boxes are drawn on the frame

        # Write the frame with bounding boxes into the output video
        out.write(frame_with_boxes)

    # Release the video writer and capture objects
    video_capture.release()
    out.release()

    return detection_video_path
