import os
from ultralytics import YOLO

# set directory
ROOT_DIR = "/home/jm/Linux/workspace/YOLOv8"  # Abosolut directory 
DATASETS = ROOT_DIR+"/Lib_datasets/stone.v10i.clip"  # the path of datasets

# Load a model
# model = YOLO("yolov8x-cls.yaml")  # build a new model from scratch
model = YOLO(ROOT_DIR+"/pre_models/yolov8x-cls.pt")  # load a pretrained model (recommended for training)

# Train the model
model.train(data=DATASETS, epochs=50, imgsz=640)

# Validate the model
metrics = model.val()  # Validate the model
metrics.top1   # top1 accuracy
metrics.top5   # top5 accuracy

# Predict with the model
model = YOLO(ROOT_DIR+"/runs/classify/train/weights/best.pt")  # load a new trained model
files = os.listdir(DATASETS+"/test/")
for file in files:   # Iterate over files in the directory
    if os.path.isdir(DATASETS+"/test/"+file+"/"):  # Check if the file is a directory
        sub_files = DATASETS+"/test/"+file+"/"  # Get the list of all files in a directory
        results = model(sub_files, save=True, save_txt=True)  # predict on images in sub_files directory

# Export the model
# model.export(format="onnx")
# model.fuse()
# model.info(verbose=True)  $ Print model information
