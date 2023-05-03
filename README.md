# chickenvision-server

This repository will contain the source code for the server or cloud service used to process the camera images and replace faces with the chicken helmet. This repository should include all the necessary files, such as Python, Node.js, or Ruby code, as well as any configuration files and dependencies.

### Download Required File

Before running the project, download the `yolov7-w6-pose.pt` file from [this link](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt) and place it in the `image-processing/yolo` directory.

### Run pose estimation manually
```bash
conda create -n yolov7_pose python=3.9
```

```bash
conda activate yolov7_pose
```

```bash
pip install -r requirements.txt
```

---

```bash
python detect.py --weights yolov7-w6-pose.pt --kpt --hide-labels --hide-conf --source image.jpg
```

```bash
python detect.py --weights yolov7-w6-pose.pt --kpt --hide-labels --hide-conf --source vid.mp4 --view-img
```