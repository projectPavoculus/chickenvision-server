from detectron2.utils.logger import setup_logger
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
setup_logger()
# import some common libraries
import numpy as np
import tqdm
import cv2
import torch
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
import time

# Extract video properties
video = cv2.VideoCapture('crowd.mp4')
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = video.get(cv2.CAP_PROP_FPS)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)
# Initialize video writer
video_writer = cv2.VideoWriter('out.mp4', fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(frames_per_second), frameSize=(width, height), isColor=True)
from matplotlib.patches import Rectangle

v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE)

def draw_boxes(frame, predictions):
    # Convert Matplotlib RGB format to OpenCV BGR format
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Get the height and width of the image
    height, width, _ = frame.shape

    # Loop over all predictions in the current frame
    for prediction in predictions:
        # Get the predicted class label and the bounding box coordinates
        class_label = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[prediction["pred_classes"][0]]
        xmin, ymin, xmax, ymax = prediction["pred_boxes"].tensor[0].cpu().numpy()

        # Convert the bounding box coordinates to pixel values
        xmin = int(xmin * width)
        ymin = int(ymin * height)
        xmax = int(xmax * width)
        ymax = int(ymax * height)

        # Draw the bounding box with a transparent color
        rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='g', facecolor='none', alpha=0.5)
        plt.gca().add_patch(rect)

        # Add the predicted class label to the top-left corner of the bounding box
        plt.gca().text(xmin, ymin, class_label, fontsize=8, color='g', bbox=dict(facecolor='none', edgecolor='g', alpha=0.5))

    return frame

# Initialize visualizer
v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE)

def runOnVideo(video, maxFrames):
    """ Runs the predictor on every frame in the video (unless maxFrames is given),
    and returns the frame with the predictions drawn.
    """

    frame_count = 0
    while True:
        # Read the next frame from the video
        has_frame, frame = video.read()

        # Exit the loop if we have reached the end of the video
        if not has_frame:
            break

        # Get the predictions for the current frame
        outputs = predictor(frame)

        # Draw the bounding boxes on the current frame
        frame = draw_boxes(frame, outputs["instances"].to("cpu"))

        # Write the current frame to the output video file
        video_writer.write(frame)

        # Increment the frame count
        frame_count += 1

        # Stop processing frames if we have processed the maximum number of frames
        if frame_count > maxFrames:
            break

# Create a cut-off for debugging
    def count_frames(video_path):
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
        return total_frames
    
    num_frames = count_frames(video)

# Enumerate the frames of the video
for visualization in tqdm.tqdm(runOnVideo(video, num_frames), total=num_frames):

    # Write test image
    cv2.imwrite('POSE detectron2.png', visualization)

    # Write to video file
    video_writer.write(visualization)

# Release resources
video.release()
video_writer.release()
cv2.destroyAllWindows()