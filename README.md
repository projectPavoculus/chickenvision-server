# YOLOv8 Pose Estimation 🐔📦

A guide on how to setup and run pose estimation using YOLOv8 in the Chickenvision project.

## Setup 🛠️

### Step 1: Download the Data Files
In the provided release, there are three `.pt` files available for download. Click on each file to initiate the download process to your local machine.

### Step 2: Locate the Downloaded Files
Once the download completes, locate the `.pt` files in your system's default download directory (usually the "Downloads" folder).

### Step 3: Move the Files to the YOLOv8 Directory
Navigate to the `Yolov8` directory in your project. This is the designated folder for YOLOv8-related files. Transfer the downloaded `.pt` files from their current location to this `Yolov8` folder.

## Run Pose Estimation 🏃‍♀️

### Step 1: Open Anaconda Terminal
Open an Anaconda terminal. Navigate to the `chickenvision-server/Yolov8` directory. Please note that Anaconda terminal is not the same as the PowerShell terminal.

```bash
cd .../chickenvision-server/Yolov8
```

### Step 2: Create a new Conda Environment
Create a new Conda environment using Python 3.9:

```bash
conda create -n yolov8 python=3.9
```

### Step 3: Activate the Conda Environment
Activate the newly created Conda environment:

```bash
conda activate yolov8
```

### Step 4: Install Required Python Packages
Install the necessary Python packages specified in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage 🚀

### Run Pose Estimation on a Video
To run pose estimation on a video, follow these steps:

1. Modify the `Yolov8/tempKeypoint.py` file to change the input `.mp4` file for processing.
2. Execute the keypoint detection and extraction script:

```bash
python tempKeypoint.py
```

## License 📚

This project uses licenses from other repositories. Please ensure to adhere to the licensing requirements of the original repositories and provide appropriate credits.
