import csv
import cv2
import os
import torch
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor


class PosePredictor(DetectionPredictor):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        super().__init__(cfg, overrides)
        self.args.task = 'pose'

    def postprocess(self, preds, img, orig_img):
        results = []
        frame_index = 0

        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

            if hasattr(self.model, 'num_keypoints'):
                num_keypoints = self.model.num_keypoints
            else:
                num_keypoints = 0

            if num_keypoints > 0:
                if hasattr(self.model, 'num_coords'):
                    num_coords = self.model.num_coords
                else:
                    num_coords = (pred.shape[1] - 6) // num_keypoints

                pred_kpts = pred[:, 6:].view(-1, num_keypoints, num_coords)
            else:
                pred_kpts = torch.empty((0, 0, 0), dtype=torch.float32)


            if pred_kpts.numel() > 0:
                pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, shape)
            else:
                pred_kpts = torch.empty((0, 0, 0), dtype=torch.float32)
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            result = Results(orig_img=orig_img,
                            path=img_path,
                            names=self.model.names,
                            boxes=pred[:, :6],
                            keypoints=pred_kpts)
            results.append(result)

            # Save keypoint detections to a file
            output_file = "keypoints_output.csv"
            with open(output_file, 'a', newline='') as file:
                writer = csv.writer(file)
                if os.stat(output_file).st_size == 0:
                    writer.writerow(["Frame", "Person", "Keypoint", "X", "Y"])

                for j, detection in enumerate(pred):
                    keypoints = detection[6:].tolist()
                    for k, keypoint in enumerate(keypoints):
                        writer.writerow([frame_index, j, k, keypoint[0], keypoint[1]])

            frame_index += 1

        return results



def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO to predict objects in an image or video."""
    model = cfg.model or 'yolov8n-pose.pt'
    source = cfg.source if cfg.source is not None else 'uni.mp4'  # modify this to point to your video file

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = PosePredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict()