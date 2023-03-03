import torch
import gc
from ultralytics import YOLO

# set directory
ROOT_DIR = "/home/jm/Linux/workspace/YOLOv8"  # Abosolut directory
DATASETS=ROOT_DIR+"/Lib_datasets/stone.v10i.yolov8"  # Don't change its name best.

# Load a model
##  model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO(ROOT_DIR+"/pre_models/yolov8s-seg.pt")  # load a pretrained model (recommended for training)

# Train the model
model.train(data=DATASETS+"/data.yaml", epochs=50, imgsz=640)

# Validate the model
metrics = model.val()  # Validate the model
metrics.box.map    # map50-95(B)
metrics.box.map50  # map50(B)
metrics.box.map75  # map75(B)
metrics.box.maps   # a list contains map50-95(B) of each category
metrics.seg.map    # map50-95(M)
metrics.seg.map50  # map50(M)
metrics.seg.map75  # map75(M)
metrics.seg.maps   # a list contains map50-95(M) of each category
del metrics
gc.collect()

# Predict with the model
model=YOLO(ROOT_DIR+"/runs/segment/train/weights/best.pt") # load a new trained model
results = model(DATASETS+"/test/images/",device=torch.device('cpu'), save=True, save_txt=True)  # predict on an image

# Export the model
# model.export(format="onnx")
# model.fuse()
# model.info(verbose=True)  # Print model information
