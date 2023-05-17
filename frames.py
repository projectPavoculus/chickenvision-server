import cv2

def count_frames(video_path):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return total_frames

# Example usage
video_path = 'uni.mp4'
num_frames = count_frames(video_path)
print("Number of frames:", num_frames)