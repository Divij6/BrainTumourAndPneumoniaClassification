

# from roboflow import Roboflow
# rf = Roboflow(api_key="7dcNBe2Nt3HsL04vJEOw")
# project = rf.workspace("silo-rvgms").project("silo-detection")
# version = project.version(3)
# dataset = version.download("yolov11")

from ultralytics import YOLO
model = YOLO("yolo11n.pt")  # Load the YOLO model
# model.train(data="datasets/Silo-detection-3/data.yaml", epochs=100, imgsz=640, batch=16, cache=False, resume=False, pretrained=False)
model.val(data="datasets/Silo-detection-3/data.yaml", cache=False)

          