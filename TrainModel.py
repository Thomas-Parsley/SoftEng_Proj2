"""
TrainModel.py

This script trains a YOLOv8 model on a custom dataset using the Ultralytics library.
It loads a pretrained YOLO model, applies training parameters, and saves results
into the runs/train/ directory.

Usage:
    python TrainModel.py
"""

from ultralytics import YOLO


def main():
    """
    Trains a YOLOv8 model using parameters defined in this script.

    - Loads a YOLOv8x pretrained model
    - Trains on the dataset defined by dataset.yaml
    - Saves results in runs/train/yolo_x_model

    Returns:
        None
    """
    # Load YOLOv8 pretrained model
    # model = YOLO("yolov8n.pt")  # small, fast; can use yolov8s.pt, yolov8m.pt for better accuracy
    model = YOLO("yolov8x.pt")

    # Train on your dataset
    model.train(
        data="dataset.yaml",  # path to a YAML file defining train/val folders and class names
        epochs=50,            # number of training epochs
        imgsz=512,            # image size
        batch=16,             # batch size
        freeze=10,        # <-- important: preserves cat/dog/person
        lr0=0.0005,       # lower LR prevents forgetting
        project="runs/train", # where to save results
        name="yolo_x_model",    # folder name for this experiment
        #workers=0
    )

if __name__ == "__main__":
    main()