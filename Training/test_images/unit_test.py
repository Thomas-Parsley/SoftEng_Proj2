from ultralytics import YOLO
import cv2
import os

updates = {"person":0, "cat":1, "dog":2, "dalek":3, "lightsaber":4}

model = YOLO("../runs/detect/dect_model_test_results21/weights/best.pt")

def test_object(object_name):
    accuracy_count = 0
    dir_path = "./test_images" + "/" + str(updates[object_name]) + object_name
    os.chdir(dir_path)
    # print("accuracy count for", object_name, ":", accuracy_count)
    for filename in os.listdir():
        if filename.endswith(".jpg") == True or filename.endswith(".png") == True:
            img = cv2.imread(filename)
            result = model(img, verbose=False)
            annotated = result[0].plot()
            for im_cls in result[0].boxes.cls:
                if int(im_cls) == updates[object_name]:
                    accuracy_count += 1
                    break
            cv2.imwrite("annotated/annotated_" + filename, annotated)
    num_files = len(os.listdir()) - 1  # subtract 1 for annotated folder
    os.chdir("../..")
    print("accuracy count for", object_name, ":", accuracy_count, num_files)
    return (accuracy_count / num_files) * 100
    

def basic_test():
    img_name = "000000000665.jpg"
    img_path = "./test_images/initial/" + img_name
    img = cv2.imread(img_path)
    result = model(img)
    annotated = result[0].plot()
    cv2.imwrite("test_images/initial/annotated/annotated_" + img_name, annotated)

if __name__ == "__main__":
    p_a = test_object("person")
    p_c = test_object("cat")
    p_d = test_object("dog")
    p_k = test_object("dalek")
    p_l = test_object("lightsaber")

    print("Person detection accuracy:", p_a)
    print("Cat detection accuracy:", p_c)
    print("Dog detection accuracy:", p_d)
    print("Dalek detection accuracy:", p_k)
    print("Lightsaber detection accuracy:", p_l)