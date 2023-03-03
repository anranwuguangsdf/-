from ultralytics import YOLO

# set directory
ROOT_DIR = "."

# Load a model
# model = YOLO(ROOT_DIR+"/pre_models/yolov8n.pt")  # load an offical detection model
model = YOLO(ROOT_DIR+"/pre_models/yolov8s-seg.pt")  # load an offical segmentation model
# model = YOLO("path/to/best.pt")  # load a custom model

# Track with the model
results = model.track(source="https://youtu.be/Zgi9g1ksQHc", show=True)  # default tracker="botsort.yaml"
# results = model.track(source="https://youtu.be/Zgi9g1ksQHc", conf=0.3, iou=0.5, show=True, tracker="bytetrack,yaml")
