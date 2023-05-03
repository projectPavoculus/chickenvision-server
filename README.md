# 🐔 Chickenvision-Server

This repository contains the source code for the server or cloud service used to process camera images and replace faces with chicken helmets. The repository includes all necessary files, such as Python, Node.js, or Ruby code, as well as any configuration files and dependencies.

## 📦 Setup

Before running the project, follow the steps below:

### 1️⃣ Download Required File

Download the `yolov7-w6-pose.pt` file from [this link](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt) and place it in the `image-processing/yolo` directory.

### 2️⃣ Run Pose Estimation Manually

Open a terminal and navigate to the `image-processing/yolo` directory:

```bash
cd .../chickenvision-server/image-processing/yolo
```

Create a new Conda environment with Python 3.9:

```bash
conda create -n yolov7_pose python=3.9
```

Activate the Conda environment:

```bash
conda activate yolov7_pose
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

Run pose estimation on an image:

```bash
python detect.py --weights yolov7-w6-pose.pt --kpt --hide-labels --hide-conf --source image.jpg
```

Run pose estimation on a video and view the output:

```bash
python detect.py --weights yolov7-w6-pose.pt --kpt --hide-labels --hide-conf --source vid.mp4 --view-img
```

## 📚 License

Please ensure you follow the licensing requirements of the original repositories and give proper credit as mentioned in the previous answers.