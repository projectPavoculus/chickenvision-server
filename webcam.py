import cv2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Set the parameter to the camera index (0 or 1) for multiple cameras

# Initialize the predictor
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)

# Get the metadata for visualization
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

# Set the desired frame rate (in frames per second)
desired_fps = 30

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Failed to open webcam")
    exit()

# Set the frame rate of the webcam
cap.set(cv2.CAP_PROP_FPS, desired_fps)

# Print the actual frame rate set on the webcam
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print("Actual webcam frame rate:", actual_fps)

# Start capturing and displaying frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame is read successfully
    if not ret:
        print("Failed to read frame")
        break

    # Perform object detection on the frame
    outputs = predictor(frame)

    # Visualize the predictions on the frame
    v = Visualizer(frame, metadata=metadata, scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Get the visualized frame
    vis_frame = v.get_image()

    # Display the frame
    cv2.imshow("Webcam", vis_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
