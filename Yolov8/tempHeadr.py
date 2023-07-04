import csv
from ultralytics import YOLO
from ast import literal_eval
import cv2
import dlib
import os.path
import math

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
                    z = float(literal_eval(row[5]))
                    keypoints.append((frame, person, keypoint, x, y, z))

            return keypoints
    except StopIteration:
        return []

def calculate_head_rotation(image, landmarks):
    # Assuming landmarks include eye positions (e.g., landmarks[36] and landmarks[45])
    eye_left = landmarks[36]
    eye_right = landmarks[45]

    # Calculate the angle between the eyes (yaw)
    angle_yaw = -1 * dlib.angle_between_points(eye_left, eye_right)

    # Calculate the angle between the eyes and the nose (pitch)
    eye_nose = (landmarks[27][0], landmarks[27][1])
    angle_pitch = -1 * dlib.angle_between_points(eye_left, eye_nose)

    # Calculate the angle between the nose and the mouth (roll)
    nose_mouth = (landmarks[33][0], landmarks[33][1])
    angle_roll = -1 * dlib.angle_between_points(eye_nose, nose_mouth)

    # Convert the angles from radians to degrees
    angle_yaw = math.degrees(angle_yaw)
    angle_pitch = math.degrees(angle_pitch)
    angle_roll = math.degrees(angle_roll)

    return angle_yaw, angle_pitch, angle_roll

model = YOLO("yolov8n-pose.pt")

source = "uni.mp4"
results = model.predict(source=source, save=False, conf=0.5, save_txt=False, show=True)


def save_keypoints(results, output_file):
    with open(output_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if os.stat(output_file).st_size == 0:
            writer.writerow(["Frame", "Person", "Keypoint", "X", "Y", "Z"])
        for i, result in enumerate(results):
            for j, keypoints in enumerate(result.keypoints):
                head_rotation = (0, 0, 0)
                if len(keypoints) >= 68:
                    landmarks = [(int(point[0]), int(point[1])) for point in keypoints[:68]]
                    head_rotation = calculate_head_rotation(result.pred, landmarks)
                writer.writerow([i, j, 0, keypoints[0][0], keypoints[0][1], head_rotation[0], head_rotation[1], head_rotation[2]])
                file.flush()
    return results

# Rest of your code...

# Update the function call
output_file = "keypoint_output2.csv"
save_keypoints(results, output_file)

# Parse the keypoints CSV file
parsed_results = parse_keypoints_csv(output_file)

separated_keypoints = {}
for frame, person, keypoint, x, y, z in parsed_results:
    if frame not in separated_keypoints:
        separated_keypoints[frame] = {}
    if person not in separated_keypoints[frame]:
        separated_keypoints[frame][person] = []
    separated_keypoints[frame][person].append((keypoint, x, y, z))

# Initialize dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Display the keypoints with head rotation information
for frame_idx, frame_keypoints in separated_keypoints.items():
    print(f"Frame {frame_idx}:")
    for person_id, keypoints_list in frame_keypoints.items():
        print(f"Person {person_id}:")
        for keypoint_idx, (keypoint, x, y, z) in enumerate(keypoints_list):
            print(f"Keypoint {keypoint_idx}: (x={x},y={y}, z={z})")
# Load the frame
frame = cv2.imread(source)

# Detect faces in the frame
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = detector(gray)

# Iterate over the detected faces
for face in faces:
    # Get the facial landmarks
    landmarks = predictor(gray, face)

    # Calculate head rotation
    head_rotation = calculate_head_rotation(frame, landmarks)

    # Display head rotation information
    print(f"Head rotation (yaw, pitch, roll): {head_rotation}")

    print()
print()