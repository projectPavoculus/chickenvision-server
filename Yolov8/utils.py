import sys
import matplotlib.pyplot as plt
import numpy as np

# List of colors to use for the graph
colors = ["#"+''.join([np.random.choice(list('0123456789ABCDEF')) for j in range(6)]) for i in range(20)]


def loadbar(current, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """Prints a progress bar to the console."""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filledLength = int(length * current // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    if current == total:
        print()


def count_lines(file):
    """Counts the number of lines in a file."""
    return sum(1 for line in file)


def object_size(obj):
    """Calculates the size of an object in kilobytes or megabytes."""
    if sys.getsizeof(obj) < 1024:
        return f'{(sys.getsizeof(obj) / (1024.0)):.2f} KB'
    else:
        return f'{(sys.getsizeof(obj) / (1024.0 * 1024.0)):.2f} MB'


def generate_graph(frame_keypoints, frame_idx):
    """Generates a graph of the keypoints."""
    plt.figure(figsize=(19.2, 10.8), dpi=100)

    for person_id, keypoints_list in frame_keypoints.items():
        x = [coord[1] for coord in keypoints_list]
        y = [coord[2] for coord in keypoints_list]

        plt.scatter(x, y, color=colors[person_id])

        for i, coord in enumerate(keypoints_list):
            plt.annotate(coord[0], (x[i], y[i]))

    plt.gca().invert_yaxis()
    
    plt.xlim([0, 1920])
    plt.ylim([1080, 0])

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    plt.savefig(f"./../graphs/frame_{frame_idx}.png")


def keypoints_origin(keypoints):
    """Returns the central point of the first 5 keypoints."""
    if len(keypoints) < 5:
        raise Exception("Not enough keypoints to calculate origin.")
    x = sum([keypoint[1] for keypoint in keypoints[:5]]) / 5
    y = sum([keypoint[2] for keypoint in keypoints[:5]]) / 5
    return x, y
