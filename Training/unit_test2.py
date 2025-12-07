from ultralytics import YOLO
import cv2
import os

updates = {"cat":0, "dalek":1, "dog":2, "human":3, "lightsaber":4, "objects":5, "sith_saber":6}
classes = ["cat", "dalek", "dog", "human", "lightsaber", "objects", "sith_saber"]
model_nums = ["22", "21",]
source_dir = "5.0/data/test"

model_1 = YOLO("../runs/detect/dect_model_test_results22/weights/best.pt")
model_2 = YOLO("../runs/detect/dect_model_test_results21/weights/best.pt")
# model_3 = YOLO("../runs/detect/dect_model_test_results24/weights/best.pt")

models = [model_1, model_2]

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
    if not os.path.exists("unit_tests/5.0/" + model_num):
        os.mkdir("unit_tests/5.0/" + model_num)
    if not os.path.exists("unit_tests/5.0/" + model_num + "/" + object_name):
        os.mkdir("unit_tests/5.0/" + model_num + "/" + object_name)
    for filename in test_files:
        if filename.endswith(".jpg") == True or filename.endswith(".png") == True:
            img = cv2.imread(dir_path + "/" + filename)
            result = model(img, verbose=False)
            annotated = result[0].plot()
            for im_cls in result[0].boxes.cls:
                if int(im_cls) == updates[object_name]:
                    accuracy_count += 1
                    break
            cv2.imwrite("unit_tests/5.0/" + model_num + "/" + object_name + "/" + filename, annotated)
    num_files = len(test_files)
    # print("accuracy count for", object_name, ":", accuracy_count, num_files)
    return (accuracy_count / num_files) * 100
    

def basic_test():
    img_name = "000000000665.jpg"
    img_path = "./test_images/initial/" + img_name
    img = cv2.imread(img_path)
    result = model(img)
    annotated = result[0].plot()
    cv2.imwrite("test_images/initial/annotated/annotated_" + img_name, annotated)

if __name__ == "__main__":
    for model_ in models:
        model_num = model_nums[models.index(model_)]
        print("Testing model number:", model_num)
        for object_ in classes:
            if object_ != "objects":
                print("Testing object:", object_)
                accuracy = test_object(object_, model_, imgs_to_test(object_), model_num)
                print(f"Accuracy for {object_} with model {model_num}: {accuracy}%")

    # test_object("cat", model_1, imgs_to_test("cat"), "22")
    # p_a = test_object("person")
    # p_c = test_object("cat")
    # p_d = test_object("dog")
    # p_k = test_object("dalek")
    # p_l = test_object("lightsaber")

    # print("Person detection accuracy:", p_a)
    # print("Cat detection accuracy:", p_c)
    # print("Dog detection accuracy:", p_d)
    # print("Dalek detection accuracy:", p_k)
    # print("Lightsaber detection accuracy:", p_l)