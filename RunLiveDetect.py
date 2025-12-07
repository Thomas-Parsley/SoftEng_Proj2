"""
RunLiveDetect.py

This script performs object recognition in real time.
It accesses the webcam and uses a trained model to annotate objects and
continuously display the results.

Usage:
    python RunLiveDetect.py
"""

from ultralytics import YOLO
import cv2

def main():
    """
    Runs object recognition model on webcam feed.

    - Loads our trained YOLO model
    - Annotates live webcam video
    - Displays annotated video

    Returns:
        None
    """
    # Load your trained model
    model = YOLO("runs/detect/dect_model_test_results22/weights/best.pt")
    #model = YOLO("yolov8x.pt")

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        count = 1
        while not cap.isOpened() and count < 9:
            print("Error: Could not open webcam. Retrying.")
            cap = cv2.VideoCapture(count)
            count += 1
        if not cap.isOpened():
            print("Error: Could not open webcam. Giving up.")
            exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Run detection
        results = model(frame)

        # Annotate frame
        annotated_frame = results[0].plot()

        # Display the frame
        cv2.imshow("YOLO Webcam Detection", annotated_frame)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam detection stopped.")

if __name__ == "__main__":
    main()