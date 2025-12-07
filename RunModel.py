"""
RunModel.py

This script performs object recognition on a video.
It reads in a .mp4 video and uses a trained model to annotate objects and
output the annotated video to another .mp4 file.

Usage:
    python RunModel.py
"""

from ultralytics import YOLO
import cv2

def main(model_num):
    """
    Runs object recognition model on .mp4 video.

    - Loads our trained YOLO model
    - Loads input .mp4 video
    - Annotates video and saves

    Returns:
        None
    """
    # Load your trained model
    model_path = "runs/detect/dect_model_test_resultsxx/weights/best.pt"
    model = YOLO(model_path.replace("xx", model_num))
    #model = YOLO("yolov8x.pt")

    # Input & output videos
    input_video = input("Enter path to input video (e.g., video_tests/inputs/cat.mp4): ")
    output_video = input("Enter path to output video (e.g., video_tests/outputs/annotated_cat.mp4): ")

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

if __name__ == "__main__":
    main("22")