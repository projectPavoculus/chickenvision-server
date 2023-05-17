import numpy as np
import cv2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog


class Detector:
    def __init__(self, model_type):
        self.cfg = get_cfg()
        self.model_type = model_type
        # Load model config and pretrained model
        if model_type == "OD":  # object detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        elif model_type == "IS":  # Instant segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        elif model_type == "KP":  # keypoint
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
        elif model_type == "LVIS":  # LVIS segmentation
            self.cfg.merge_from_file(
                model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        elif model_type == "PS":  # Panoptic Segmentation
            self.cfg.merge_from_file(
                model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-PanopticSegmentation/panoptic_fpn_R_101_FPN_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.predictor = DefaultPredictor(self.cfg)
        if self.cfg.MODEL.DEVICE == "cuda":
            device = torch.device("cuda:0")
            print("Using GPU:", torch.cuda.get_device_name(device))
        else:
            print("CUDA not available")

    def onImage(self, imagePath):
        image = cv2.imread(imagePath)
        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")

        if self.model_type != "PS":
            predictions = self.predictor(image)
            viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                            instance_mode=ColorMode.IMAGE)
            output = viz.draw_instance_predictions(instances)

        else:
            predictions, segmentInfo = outputs["panoptic_seg"]
            viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
            output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)

        cv2.imshow("Results", output.get_image()[:, :, ::-1])
        cv2.waitKey(0)

    def OnVideo(self, videoPath):
        cap = cv2.VideoCapture(videoPath)

        if not cap.isOpened():
            print("Error opening the file....")
            return

        (success, image) = cap.read()
        while success:
            if self.model_type != "PS":
                predictions = self.predictor(image)
                viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                                instance_mode=ColorMode.IMAGE)
                output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

            else:
                predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
                viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
                output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)

            cv2.imshow("Results", output.get_image()[:, :, ::-1])
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            (success, image) = cap.read()