import cv2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog
import asyncio

async def process_frames(video_path, max_frames, predictor, video_writer, visualizer):
    video = cv2.VideoCapture(video_path)
    read_frames = 0

    while True:
        has_frame, frame = video.read()
        if not has_frame:
            break

        # Get prediction results for this frame
        outputs = predictor(frame)

        # Make sure the frame is colored
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw a visualization of the predictions using the video visualizer
        visualization = visualizer.draw_instance_predictions(frame, outputs["instances"].to("cpu"))

        # Convert Matplotlib RGB format to OpenCV BGR format
        visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)

        # Write to video file
        video_writer.write(visualization)

        read_frames += 1
        if read_frames >= max_frames:
            break

    video.release()

async def main():
    # Extract video properties
    video_path = 'uni.mp4'
    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    video_writer = cv2.VideoWriter('uni_A.mp4', cv2.VideoWriter_fourcc(*"mp4v"), float(frames_per_second), (width, height), isColor=True)

    # Initialize predictor
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cpu"  # switch to CPU
    predictor = DefaultPredictor(cfg)

    # Initialize visualizer
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    visualizer = VideoVisualizer(metadata, ColorMode.IMAGE)

    # Create a cut-off for debugging
    def count_frames(video_path):
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
        return total_frames
    # Create a cut-off for debugging
    num_frames = count_frames(video)

    # Start processing frames asynchronously
    await process_frames(video_path, num_frames, predictor, video_writer, visualizer)

    # Release resources
    video_writer.release()
    cv2.destroyAllWindows()

# Run the main function asynchronously
asyncio.run(main())