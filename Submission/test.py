from ultralytics import YOLO
import cv2
import os
from RunModel import main

updates = {"cat":0, "dalek":1, "dog":2, "human":3, "lightsaber":4, "objects":5, "sith_saber":6}
classes = ["cat", "dalek", "dog", "human", "lightsaber", "objects", "sith_saber"]
source_dir = "test"

model_path = "../runs/detect/dect_model_test_resultsxx/weights/best.pt"

def imgs_to_test(object_name):
    dir_path = source_dir + "/labels"
    files = os.listdir(dir_path)
    test_files = []

    for filename in files:
        with open(dir_path + "/" + filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if int(line[0]) == updates[object_name]:
                    test_files.append(filename[:-3] + "jpg")
                    break
    return test_files

def test_object(object_name, model, test_files, model_num):
    accuracy_count = 0
    dir_path = source_dir + "/images"
    if not os.path.exists("tests/object_tests" + model_num):
        os.mkdir("tests/object_tests/" + model_num)
    if not os.path.exists("tests/object_tests/" + model_num + "/" + object_name):
        os.mkdir("tests/object_tests/" + model_num + "/" + object_name)
    for filename in test_files:
        if filename.endswith(".jpg") == True or filename.endswith(".png") == True:
            img = cv2.imread(dir_path + "/" + filename)
            result = model(img, verbose=False)
            annotated = result[0].plot()
            for im_cls in result[0].boxes.cls:
                if int(im_cls) == updates[object_name]:
                    accuracy_count += 1
                    break
            cv2.imwrite("tests/object_tests/" + model_num + "/" + object_name + "/" + filename, annotated)
    num_files = len(test_files)
    return (accuracy_count / num_files) * 100 

def basic_test(model):
    for image in os.listdir(".Training/test_images/initial/"):
        if image.endswith(".jpg") == True:
            img_name = image
            img_path = ".Training/test_images/initial/" + img_name
            img = cv2.imread(img_path)
            result = model(img)
            annotated = result[0].plot()
            cv2.imwrite("test_images/initial/annotated/annotated_" + img_name, annotated)

def unit_test(model_, model_num):
    with open("tests/object_tests/" + model_num + "/results.txt", "w") as result_file:
        for object_ in classes:
            if object_ != "objects":
                result_file.write(("Testing object:", object_))
                accuracy = test_object(object_, model_, imgs_to_test(object_), model_num)
                result_file.write(f"Accuracy for {object_} with model {model_num}: {accuracy}%")

def video_test(model_num):
    main(model_num)

if __name__ == "main":
    model_num = input("Enter model number to test: ")
    model = model_path.replace("xx", model_num)
    yolo_model = YOLO(model)
    basic_test(yolo_model)
    unit_test(yolo_model, model_num)
    video_test(model_num)