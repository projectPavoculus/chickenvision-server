import csv
from tqdm import tqdm
from ultralytics import YOLO
from utils import count_lines
from keypoints_analisis import generate_graph, direction_pitch, direction_yaw, save_pitch_yaw_headsize_origin
import os

# Separate the keypoints by frame and person
def separate_keypoints(keypoints):
    separated_keypoints = {}
    for frame, person, keypoint, x, y in keypoints:
        if frame not in separated_keypoints:
            separated_keypoints[frame] = {}
        if person not in separated_keypoints[frame]:
            separated_keypoints[frame][person] = []
        separated_keypoints[frame][person].append((keypoint, x, y))
    return separated_keypoints


# Display the keypoints
def display_keypoints(keypoints, save_graph=False):
    for frame_idx, frame_keypoints in keypoints.items():
        print(f"Frame {frame_idx}:")
        if save_graph:
            generate_graph(keypoints, frame_idx)
        for person_id, keypoints_list in frame_keypoints.items():
            print(f"Person {person_id}: pitch / yaw = {direction_pitch(keypoints_list)} / {direction_yaw(keypoints_list)}")
            for keypoint, x, y in keypoints_list[:6]:
                print(f"Keypoint {keypoint}: (x={x}, y={y})")
            print()
        print()

def check_video_file(source):
    if not os.path.isfile(source):
        print(f"File '{source}' does not exist.")
        return False
    
    source_extension = os.path.splitext(source)[1].lower()
    valid_extensions = [".mp4", ".mov", ".mpeg"]

    if source_extension not in valid_extensions:
        print("File type is incorrect.")
        return False

    return True

def main():
    model = YOLO("yolov8n-pose.pt")

    video_file = "bazzar.mp4"


    if check_video_file(video_file):
        results = model.predict(source=video_file, save=False, conf=0.5, save_txt=False, show=True)
        separated_keypoints = separate_keypoints(results)
        save_pitch_yaw_headsize_origin(separated_keypoints, 'pitch_yaw_headsize.json')

    print("Done!")

if __name__ == "__main__":
    main()