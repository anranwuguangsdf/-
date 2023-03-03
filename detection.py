from ultralytics import YOLO

#set directory
ROOT_DIR = "/home/jm/Linux/workspace/YOLOv8"  # Abosolut directory
DATASETS = ROOT_DIR+"/Lib_datasets/stone.v10i.yolov8"  # Don't change its name best.

# Load a model
##  model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO(ROOT_DIR+"/pre_models/yolov8s.pt")  # load a pretrained model (recommended for training)

# Train the model
model.train(data=DATASETS+"/data.yaml", epochs=50, imgsz=640)



# Validate the model
metrics = model.val()  #IF Null, It'll automatically evaluate the data you trained.   model.val(data="coco128.yaml")
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category

# Predict with the model
model=YOLO(ROOT_DIR+"/runs/detect/train/weights/best.pt")     # load a new trained model
results = model(DATASETS+"/test/images/", save=True, save_txt=True)  # predict on an image

# Export the model
# model.export(format="onnx")
