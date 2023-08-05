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