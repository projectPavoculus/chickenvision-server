import open3d as o3d
import json
import numpy as np
import cv2
import os

# Function to extract frames from a video
def extract_frames_from_video(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_id in range(total_frames):
        success, frame = video_capture.read()
        if not success:
            break

        frame_path = os.path.join(output_dir, "frame{}.png".format(frame_id))
        cv2.imwrite(frame_path, frame)

        print("Extracted frame", frame_id)

    video_capture.release()

# Load the chicken head model
chicken_head_model = o3d.io.read_triangle_mesh("ChickenHat.obj")

# Load the JSON file with head coordinates
with open("pitch_yaw_headsize.json", "r") as json_file:
    head_data = json.load(json_file)

def render_chicken_head(frame, head_coordinates,scaling_factor):
    yaw_angle = head_coordinates["Yaw"]
    scale_factor = head_coordinates["HeadSize"] / scaling_factor

    frame_np = np.array(frame)
    chicken_head = np.ones((100,100,3), dtype=np.uint8) * 255

    x, y = head_coordinates["OriginPoint"]
    x, y = int(x), int(y)
    h, w, _ = chicken_head.shape

    frame_np[y:y+h, x:x+w] = chicken_head

    return frame_np

scaling_factor_sum = 0.0
num_frames = 0

for frame_info in head_data["Frames"]:
    for person_info in frame_info["Persons"]:
        scaling_factor_sum += person_info["HeadSize"]
        num_frames += 1

empirical_scaling_factor = scaling_factor_sum / num_frames # Calculate the average scaling factor

video_path = "bazzar.mp4"
output_dir = "output_frames"

extract_frames_from_video(video_path, output_dir)

for frame_info in head_data["Frames"]:
    frame_id = frame_info["FrameID"]

    for person_info in frame_info["Persons"]:
        person_id = person_info["PersonID"]
        head_coordinates = {
            "Yaw": person_info["Yaw"],
            "HeadSize": person_info["HeadSize"],
            "OriginPoint": person_info["OriginPoint"]
        }

        frame_path = os.path.join(output_dir, "frame{}.png".format(frame_id))
        frame = cv2.imread(frame_path)

        frame_with_chicken_head = render_chicken_head(frame, head_coordinates, empirical_scaling_factor)
        output_path = os.path.join(output_dir, "frame{}_person{}.png".format(frame_id, person_id))
        cv2.imwrite(output_path, frame_with_chicken_head)