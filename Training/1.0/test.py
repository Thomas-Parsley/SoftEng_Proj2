import torch
from ultralytics import YOLO
import torchvision
import torchaudio

def main():
    dect_model = YOLO("yolov8s.pt")
    dect_model.train(data="C:\\Users\\wagne\\Git\\SoftEng_Proj2\\Training\\data.yaml",
                    epochs=100, 
                    imgsz=512, 
                    batch=16,
                    workers=2,
                    freeze=10,        # <-- important: preserves cat/dog/person
                    name="dect_model_test_results")

if __name__ == "__main__":
    main()