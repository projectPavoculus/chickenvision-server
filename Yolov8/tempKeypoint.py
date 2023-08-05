import csv
from tqdm import tqdm
from ultralytics import YOLO
from utils import count_lines
from keypoints_analisis import generate_graph, direction_pitch, direction_yaw, save_pitch_yaw_headsize_origin


# Parse keypoints from CSV file
def parse_keypoints_csv(file_path):
    try:
        print(file_path)
        with open(file_path, 'r') as file:
            l = count_lines(file_path)
            file.seek(0)  # Reset file pointer

            reader = csv.reader(file)
            next(reader)  # Skip header row

            keypoints = []
            for i, row in tqdm(enumerate(reader), total=l, desc="Parse progress:"):
                if row:
                    frame, person, keypoint, x, y = row
                    keypoints.append((int(frame), int(person), int(keypoint), float(x), float(y)))
            return keypoints
    except Exception as e:
        print(f"Error parsing keypoints from CSV file: {e}")              


# Save keypoints to CSV file
def save_keypoints(results, output_file):
    with open(output_file, 'a', newline='') as file:
        if count_lines(output_file) > 2:
            raise Exception("The file already has the necessary.")
        writer = csv.writer(file)

        l = len(results)
        for i in tqdm(range(l), desc="Parse progress:"):
            result = results[i]
            keypoints = result.keypoints
            for j, person_keypoints in enumerate(keypoints):
                for k, keypoint in enumerate(person_keypoints):
                    x, y = keypoint[:2]
                    writer.writerow([i, j, k, x.item(), y.item()])


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


model = YOLO("yolov8n-pose.pt")

source = "bazzar.mp4"
output_file = "./keypoints_output.csv"


# Save or parse the keypoints to a CSV filex
if input("\nSave or parse the keypoints? (1/2): ") == "1":
    results = model.predict(source=source, save=False, conf=0.5, save_txt=False, show=True)
    save_keypoints(results, output_file)
else:
    parsed_results = parse_keypoints_csv(output_file)

if input("\nDisplay the keypoints? (y/n): ") == "y" and parsed_results:
    separated_keypoints = separate_keypoints(parsed_results)
    display_keypoints(separated_keypoints, save_graph=True)

if input("\nSave rotation keypoints? (y/n): ") == "y" and parsed_results:
    separated_keypoints = separate_keypoints(parsed_results)
    save_pitch_yaw_headsize_origin(separated_keypoints, 'pitch_yaw_headsize.json')


print("Done!")