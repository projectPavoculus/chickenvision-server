import csv
from ultralytics import YOLO
from ast import literal_eval

def parse_keypoints_csv(file_path):
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row

            keypoints = []
            for row in reader:
                if row:
                    frame = int(row[0])
                    person = int(row[1])
                    keypoint = int(row[2])
                    x = float(literal_eval(row[3]))
                    y = float(literal_eval(row[4]))
                    keypoints.append((frame, person, keypoint, x, y))

            return keypoints
    except StopIteration:
        return []

model = YOLO("yolov8n-pose.pt")

source = "bazzar.mp4"
results = model.predict(source=source, save=False, conf=0.5, save_txt=False, show=True)

output_file = "keypoints_output.csv"

# Save keypoints to CSV file
def save_keypoints(results, output_file):
    with open(output_file, 'a', newline='') as file:
        writer = csv.writer(file)
        for i, result in enumerate(results.xyxy):
            keypoints = result.keypoints
            for j, person_keypoints in enumerate(keypoints):
                for k, keypoint in enumerate(person_keypoints):
                    x, y = keypoint[:2]
                    writer.writerow([i, j, k, x.item(), y.item()])

# Parse the keypoints CSV file
parsed_results = parse_keypoints_csv(output_file)

separated_keypoints = {}
for frame, person, keypoint, x, y in parsed_results:
    if frame not in separated_keypoints:
        separated_keypoints[frame] = {}
    if person not in separated_keypoints[frame]:
        separated_keypoints[frame][person] = []
    separated_keypoints[frame][person].append((keypoint, x, y))

# Display the keypoints
for frame_idx, frame_keypoints in separated_keypoints.items():
    print(f"Frame {frame_idx}:")
    for person_id, keypoints_list in frame_keypoints.items():
        print(f"Person {person_id}:")
        for keypoint_idx, (keypoint, x, y) in enumerate(keypoints_list):
            print(f"Keypoint {keypoint_idx}: (x={x}, y={y})")
        print()
    print()