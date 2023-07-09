import matplotlib.pyplot as plt
import numpy as np
import math

# List of colors to use for the graph
colors = ["#"+''.join([np.random.choice(list('0123456789ABCDEF')) for j in range(6)]) for i in range(20)]
red = "#FF0000"


# Generates a graph of the keypoints.
def generate_graph(frame_keypoints, frame_idx):
    plt.figure(figsize=(19.2, 10.8), dpi=100)

    for person_id, keypoints_list in frame_keypoints.items():
        x, y = keypoints_frame(keypoints_list)

        plt.scatter(x, y, color=colors[person_id])

        for i, coord in enumerate(keypoints_list):
            plt.annotate(coord[0], (x[i], y[i]))

        offsets = [60, 100, 140]
        offset = offsets[person_id % len(offsets)]
        x_center, y_center = keypoints_origin(keypoints_list)

        plt.scatter(x_center, y_center, color=red)
        plt.annotate(f'{relative_zaxis(keypoints_list)} / {relative_xaxis(keypoints_list)}', (x_center, y_center - 40))
        plt.annotate(f'{relative_zaxis_degrees(keypoints_list)} / {relative_xaxis_degrees(keypoints_list)}', (x_center, y_center - offset))
    plt.gca().invert_yaxis()
    
    plt.xlim([0, 1920])
    plt.ylim([1080, 0])

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    plt.savefig(f"./../graphs/frame_{frame_idx}.png")


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
def relative_zaxis(keypoints):
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
def relative_xaxis(keypoints):
    y_ears_avg = (keypoints[4][2] + keypoints[3][2]) / 2
    y_nose = keypoints[0][2]
    threshold = 0.1  # 10% error tolerance

    if abs(y_nose - y_ears_avg) <= threshold * 100:
        return "center"
    elif y_nose < y_ears_avg: # Inverse because of inverted Y axis
        return "up"
    else:
        return "down"
    

# Returns the relative rotation of the helmet around the Z axis (yaw) in degrees
def relative_zaxis_degrees(keypoints):
    xl = keypoints[4][1] - keypoints[2][1]
    xr = keypoints[1][1] - keypoints[3][1]
    angle = math.atan(abs(xl - xr) / abs(xl + xr))  # arctan of the ratio of difference to sum
    return math.degrees(angle)


# Returns the relative rotation of the helmet around the X axis (pitch) in degrees
def relative_xaxis_degrees(keypoints):
    pitch_direction = relative_xaxis(keypoints)
    if pitch_direction == 'center':
        return 0

    if pitch_direction == 'up':
        # Use the ear-eye pair with the larger vertical distance
        if abs(keypoints[4][2] - keypoints[2][2]) > abs(keypoints[5][2] - keypoints[3][2]):
            dy = keypoints[4][2] - keypoints[2][2]  # Ear 1 - Eye 1
            dx = keypoints[4][1] - keypoints[2][1]  # Ear 1 - Eye 1
        else:
            dy = keypoints[5][2] - keypoints[3][2]  # Ear 2 - Eye 2
            dx = keypoints[5][1] - keypoints[3][1]  # Ear 2 - Eye 2
    else:  # 'down'
        # Use the ear-eye pair with the smaller vertical distance
        if abs(keypoints[4][2] - keypoints[2][2]) < abs(keypoints[5][2] - keypoints[3][2]):
            dy = keypoints[4][2] - keypoints[2][2]  # Ear 1 - Eye 1
            dx = keypoints[4][1] - keypoints[2][1]  # Ear 1 - Eye 1
        else:
            dy = keypoints[5][2] - keypoints[3][2]  # Ear 2 - Eye 2
            dx = keypoints[5][1] - keypoints[3][1]  # Ear 2 - Eye 2

    radians = math.atan2(dy, dx)
    degrees = math.degrees(radians)
    return degrees
