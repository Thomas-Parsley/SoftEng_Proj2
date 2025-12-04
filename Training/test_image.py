from ultralytics import YOLO
import cv2
# img1 = cv2.imread("lightsaber_00040.jpg")
# img2 = cv2.imread("lightsaber_00044.jpg")
img3 = cv2.imread("test_images/initial/chat_generated.png")

model = YOLO("../runs/detect/dect_model_test_results21/weights/best.pt")
# model = YOLO("yolov8s.pt")
# result1 = model(img1)
# result2 = model(img2)
result3 = model(img3)

# annotated1 = result1[0].plot()
# annotated2 = result2[0].plot()
annotated3 = result3[0].plot()
r = result3[0]
print(int(r.boxes.cls[0]))

# cv2.imwrite("img1.jpg", annotated1)
# cv2.imwrite("img2.jpg", annotated2)
# cv2.imwrite("test_images/img4.jpg", annotated3)