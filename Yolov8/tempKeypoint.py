import csv
from ultralytics import YOLO
from ast import literal_eval
from utils import loadbar, count_lines, object_size
from keypoints_analisis import generate_graph

# Parse keypoints from CSV file
def parse_keypoints_csv(file_path):
    try:
        print(file_path)
        with open(file_path, 'r') as file:
            l = count_lines(file)
            file.seek(0)  # Reset file pointer

            reader = csv.reader(file)
            next(reader)  # Skip header row

            keypoints = []
            for i, row in enumerate(reader):
                loadbar(i+1, l, prefix='Parse progress:', suffix=f"{object_size(file)}", length=100)
                if row:
                    frame = int(row[0])
                    person = int(row[1])
                    keypoint = int(row[2])
                    x = float(literal_eval(row[3]))
                    y = float(literal_eval(row[4]))
                    keypoints.append((frame, person, keypoint, x, y))

            return keypoints
    except Exception as e:
        print(f"Error parsing keypoints from CSV file: {e}")

# Save keypoints to CSV file
def save_keypoints(results, output_file):
    with open(output_file, 'a', newline='') as file:
        if count_lines(file) > 2:
            raise Exception("The file already has the necessary.")
        writer = csv.writer(file)

        l = len(results)
        for i, result in enumerate(results):
            loadbar(i+1, l, prefix='Save progress:', suffix=f'{object_size(file)}', length=100)
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
            generate_graph(frame_keypoints, frame_idx)
        for person_id, keypoints_list in frame_keypoints.items():
            print(f"Person {person_id}:")
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

print("Done!")