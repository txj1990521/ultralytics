from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-seg.pt")  # load an official model
# Predict with the model
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

results = model("data/video/workman.mp4", save=True)  # predict on an image

