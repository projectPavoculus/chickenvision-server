import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

# List of colors to use for the graph
colors = ["#"+''.join([np.random.choice(list('0123456789ABCDEF')) for j in range(6)]) for i in range(20)]


# Generates a graph of the keypoints.
def generate_graph(frame_keypoints, frame_idx):
    plt.figure(figsize=(19.2, 10.8), dpi=100)

    for person_id, keypoints_list in frame_keypoints[frame_idx].items():
        x, y = keypoints_frame(keypoints_list)

        # Get the keypoints for the ears (indices 3 and 4) and calculate the middle point between them
        mid_ear_x, mid_ear_y = calculate_middle_point(keypoints_list[3][1:], keypoints_list[4][1:])

        # Get the keypoints for the eyes (indices 1 and 2) and calculate the middle point between them
        mid_eye_x, mid_eye_y = calculate_middle_point(keypoints_list[1][1:], keypoints_list[2][1:])

        # Calculate the head size based on the distance between the nose and the middle ear position
        nose = keypoints_list[0][1:]
        head_size = calculate_size(keypoints_list[1:5])

        # Annotate the keypoints
        for i, (xi, yi) in enumerate(zip(x, y)):
            plt.annotate(str(i), (xi, yi))

        # Plot the keypoints
        plt.scatter(x, y, color=colors[person_id])
        plt.scatter(mid_ear_x, mid_ear_y, color='blue')
        plt.scatter(mid_eye_x, mid_eye_y, color='purple')

        offsets = [80, 130, 180]
        offset = offsets[person_id % len(offsets)]
        x_center, y_center = keypoints_origin(keypoints_list)

        plt.scatter(x_center, y_center, color='red')
        plt.annotate(f'head size: {head_size:.2f}', (x_center, y_center - offset))
        plt.annotate(f'pitch: {direction_pitch(keypoints_list)} {calculate_pitch_rotation(keypoints_list, 1):.2f}', (x_center, y_center - offset - 40))
        plt.annotate(f'yaw: {direction_yaw(keypoints_list)} {calculate_yaw_rotation(keypoints_list):.2f}', (x_center, y_center - offset - 20))

    plt.gca().invert_yaxis()
    
    plt.xlim([0, 1920])
    plt.ylim([1080, 0])

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    plt.title(f'Frame {frame_idx}')
    plt.show()


# Function to calculate the size of the head based on the furthest points of the head's center
def calculate_size(keypoints):
    # Extract the x and y coordinates from the keypoints
    x_coords = [keypoint[1] for keypoint in keypoints]
    y_coords = [keypoint[2] for keypoint in keypoints]

    # Calculate the width and height of the head
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)

    # Return a single generalized size value (geometric mean of width and height)
    size = (width * height) ** 0.5
    return size


# Function to calculate the middle point between 2+ points
def calculate_middle_point(*points):
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))


# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5


# Returns the relative helmet origin point
def keypoints_origin(keypoints):
    if len(keypoints) < 5:
        raise Exception("Not enough keypoints to calculate origin.")
    x = sum([keypoint[1] for keypoint in keypoints[:5]]) / 5
    y = sum([keypoint[2] for keypoint in keypoints[:5]]) / 5
    return x, y


# Returns the relative helmet origin point
def keypoints_frame(keypoints):
    x = [coord[1] for coord in keypoints]
    y = [coord[2] for coord in keypoints]
    return x, y


# Returns the relative direction of the helmet to the Z Axis
def direction_yaw(keypoints):
    xl = keypoints[4][1] - keypoints[2][1]
    xr = keypoints[1][1] - keypoints[3][1]
    threshold = 0.3  # 30% error tolerance

    if abs(xl - xr) / max(abs(xl), abs(xr)) <= threshold:
        return "center"
    elif xl > xr:
        return "left"
    else:
        return "right"


# Returns the relative direction of the helmet to the X Axis
def direction_pitch(keypoints):
    y_ears_avg = (keypoints[4][2] + keypoints[3][2]) / 2
    y_nose = keypoints[0][2]
    threshold = 0.1  # 10% error tolerance

    if abs(y_nose - y_ears_avg) <= threshold * 100:
        return "center"
    elif y_nose < y_ears_avg: # Inverse because of inverted Y axis
        return "up"
    else:
        return "down"
    

def determine_rotation(keypoints, angle):
    direction = direction_pitch(keypoints)
    if direction == "up":
        angle *= -1  # Negative for upward rotation
    elif direction == "down":
        angle *= 1   # Positive for downward rotation
    else: # "center"
        angle = 0
    return angle


# Returns the relative rotation of the helmet around the Z axis (yaw) in degrees
def calculate_yaw_rotation(keypoints):
    xl = keypoints[4][1] - keypoints[2][1]
    xr = keypoints[1][1] - keypoints[3][1]
    angle = math.atan(abs(xl - xr) / abs(xl + xr))  # arctan of the ratio of difference to sum
    angle = math.degrees(angle)

    # Use the direction_yaw function to determine the sign of the rotation
    direction = direction_yaw(keypoints)
    if direction == "left":
        angle = 1 * angle + 19  # Negative for left rotation (inverted due to inverted Y axis)
    elif direction == "right":
        angle = -1 * angle + 19   # Positive for right rotation (i)
    else: # "center"
        angle = 0
    return angle

    
# Returns the relative rotation of the helmet around the X axis (pitch) in degrees
def calculate_pitch_rotation(keypoints, version = 0):
    def calculate_pitch_rotation_v1():
        # This version calculates the angle between the line from nose to mid-point of ears and the horizontal line (X-axis)
        mid_ear_x, mid_ear_y = calculate_middle_point(keypoints[3][1:], keypoints[4][1:])
        nose = keypoints[0][1:]
        angle = math.atan(abs(mid_ear_y - nose[1]) / abs(mid_ear_x - nose[0]))
        angle = math.degrees(angle)
    
        direction = direction_pitch(keypoints)
        if direction == "up":
            angle = 90 - angle  # Adjust for upward rotation
        elif direction == "down":
            angle = -(90 - angle)   # Adjust for downward rotation
        else: # "center"
            angle = 0
    
        return angle

    def calculate_pitch_rotation_v2():
        # This version calculates the angle between the line from left ear to right ear and the horizontal line (X-axis)
        leftear = keypoints[3][1:]
        rightear = keypoints[4][1:]
        angle = math.atan(abs(leftear[1] - rightear[1]) / abs(leftear[0] - rightear[0]))
        angle = math.degrees(angle)
    
        direction = direction_pitch(keypoints)
        if direction == "up":
            angle = 90 - angle  # Adjust for upward rotation
        elif direction == "down":
            angle = -(90 - angle)   # Adjust for downward rotation
        else: # "center"
            angle = 0

        return angle

    def calculate_pitch_rotation_v3():
        # This version calculates the average of the angles obtained from v1 and v2
        angle = (calculate_pitch_rotation_v1(keypoints) + calculate_pitch_rotation_v2(keypoints)) / 2
        return angle



    # Use the selected version to calculate pitch rotation
    if version == 1:
        return calculate_pitch_rotation_v1()
    elif version == 2:
        return calculate_pitch_rotation_v2()
    elif version == 3:
        return calculate_pitch_rotation_v3()
    else:
        raise ValueError("Invalid version number. Please select 1, 2, or 3.")
