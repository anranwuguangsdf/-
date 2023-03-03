import torch
from ultralytics import YOLO

# set directory
ROOT_DIR = "/home/jm/Linux/workspace/YOLOv8"  # Abosolut directory
DATASETS=ROOT_DIR+"/Lib_datasets/WeChat_20230303085149.mp4"  # Don't change its name best.

# Predict with the model
#model=YOLO(ROOT_DIR+"/runs/segment/train/weights/best.pt") # load a new trained model
model=YOLO(ROOT_DIR+"/pre_models/yolov8n-seg.pt") # load a new trained model
results = model(DATASETS,device=torch.device('cpu'), save=True, show=True)  # predict on an image

# Export the model
# model.export(format="onnx")
# model.fuse()
# model.info(verbose=True)  # Print model information
