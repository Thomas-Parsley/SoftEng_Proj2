import torch
from ultralytics import YOLO
import torchvision
import torchaudio


def main():
    dect_model1 = YOLO("yolov8s.pt")
    dect_model1.train(data="C:\\Users\\wagne\\Git\\SoftEng_Proj2\\Training\\5.0\\data.yaml",
                    epochs=100, 
                    imgsz=512, 
                    batch=16,
                    workers=2,
                    freeze=0,        # <-- important: preserves cat/dog/person
                    name="dect_model_test_results")
    dect_model2 = YOLO("yolov8s.pt")
    dect_model2.train(data="C:\\Users\\wagne\\Git\\SoftEng_Proj2\\Training\\5.0\\data.yaml",
                    epochs=100, 
                    imgsz=512, 
                    batch=16,
                    workers=2,
                    freeze=12,        # <-- important: preserves cat/dog/person
                    name="dect_model_test_results")
    dect_model3 = YOLO("yolov8s.pt")
    dect_model3.train(data="C:\\Users\\wagne\\Git\\SoftEng_Proj2\\Training\\5.0\\data.yaml",
                    epochs=100, 
                    imgsz=512, 
                    batch=16,
                    workers=2,
                    freeze=22,        # <-- important: preserves cat/dog/person
                    name="dect_model_test_results")

if __name__ == "__main__":
    main()