from ultralytics import YOLO

# Load YOLOv8 pretrained model
model = YOLO("yolov8n.pt")  # small, fast; can use yolov8s.pt, yolov8m.pt for better accuracy

# Train on your dataset
model.train(
    data="dataset.yaml",  # path to a YAML file defining train/val folders and class names
    epochs=50,            # number of training epochs
    imgsz=640,            # image size
    batch=16,             # batch size
    project="runs/train", # where to save results
    name="custom_yolo"    # folder name for this experiment
)
