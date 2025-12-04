import fiftyone as fo
import fiftyone.zoo as foz

# dataset1 = foz.load_zoo_dataset("coco-2017", label_types="detections", classes=["person"], max_samples=100)
# dataset2 = foz.load_zoo_dataset("coco-2017", label_types="detections", classes=["cat"], max_samples=100)
dataset3 = foz.load_zoo_dataset("coco-2017", label_types="detections", classes=["dog"], max_samples=100)