from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("runs/train/custom_yolo/weights/best.pt")

# Input & output videos
input_video = "input.mp4"
output_video = "output_annotated.mp4"

# Open video
cap = cv2.VideoCapture(input_video)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Annotate frame (YOLOv8 built-in plotting)
    annotated_frame = results[0].plot()

    # Write frame to output
    out.write(annotated_frame)

cap.release()
out.release()
print("Finished video annotation!")
