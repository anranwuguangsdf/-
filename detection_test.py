from ultralytics import YOLO

#set directory
ROOT_DIR = "/home/jm/Linux/workspace/YOLOv8"  # Abosolut directory
DATASETS = ROOT_DIR+"/Lib_datasets/WeChat_20230303085149.mp4"  # Don't change its name best.

# Predict with the model
model=YOLO(ROOT_DIR+"/runs/detect/train/weights/best.pt")     # load a new trained model
#model=YOLO(ROOT_DIR+"/pre_models/yolov8s.pt")     # load a new trained model
results = model(DATASETS, save=True, save_txt=True)  # predict on an image

# Export the model
# model.export(format="onnx")
