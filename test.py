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

        frame_path = os.path.join(output_dir, "frame{:04d}.png".format(frame_id))
        cv2.imwrite(frame_path, frame)

        print("Extracted frame", frame_id)

    video_capture.release()
#^ this works

# Load the chicken head model
file_path = "ChickenHat.obj"
chicken_head_model = o3d.io.read_triangle_mesh(file_path.encode("mbcs"))

# Load the JSON file with head coordinates
with open("pitch_yaw_headsize.json", "r") as json_file:
    head_data = json.load(json_file)

def render_chicken_head(frame, head_coordinates, scaling_factor):
    yaw_angle = head_coordinates["Yaw"]
    scale_factor = head_coordinates["HeadSize"] / scaling_factor

    # Create a transparent image with the same dimensions as the frame
    transparent_layer = np.zeros_like(frame, dtype=np.uint8)

    # Render the chicken_head_model on the transparent image
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=frame.shape[1], height=frame.shape[0])
    vis.get_render_option().background_color = np.array([0, 0, 0, 0])  # Transparent background
    vis.add_geometry(chicken_head_model)
    vis.update_geometry(chicken_head_model)
    vis.poll_events()
    vis.update_renderer()

    # Set the view to match the head_coordinates
    view_ctrl = vis.get_view_control()
    view_ctrl.set_lookat([0, 0, 0])
    view_ctrl.set_zoom(scale_factor * 3)
    view_ctrl.set_front([0, 0, -1])
    view_ctrl.set_up([0, -1, 0])

    rot_mat = np.array([[np.cos(yaw_angle), 0, np.sin(yaw_angle)],
                        [0, 1, 0],
                        [-np.sin(yaw_angle), 0, np.cos(yaw_angle)]])
    vis.get_view_control().convert_from_pinhole_camera_parameters(
        [0, 0, scale_factor * 3],
        [0, 0, 0],
        [0, -1, 0]
    )

    vis.get_render_option().point_size = 3.0
    vis.update_geometry()

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image("temp_chicken_head.png")

    vis.destroy_window()

    # Load the rendered chicken_head_model with the transparent background
    chicken_head = cv2.imread("temp_chicken_head.png", cv2.IMREAD_UNCHANGED)

    # Extract the alpha channel from the rendered chicken_head_model
    chicken_alpha = chicken_head[:, :, 3]

    # Convert alpha channel to a binary mask
    chicken_binary_mask = (chicken_alpha > 0).astype(np.uint8) * 255

    # Use the binary mask to overlay the chicken_head_model on the transparent layer
    transparent_layer[y:y+chicken_head.shape[0], x:x+chicken_head.shape[1]] = cv2.bitwise_and(chicken_head[:, :, :3], chicken_head[:, :, :3], mask=chicken_binary_mask)

    # Combine the transparent layer with the frame
    frame = cv2.addWeighted(frame, 1, transparent_layer, 1, 0)

    return frame

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

        frame_path = os.path.join(output_dir, "frame{:04d}.png".format(frame_id))
        frame = cv2.imread(frame_path)

        frame_with_chicken_head = render_chicken_head(frame, head_coordinates, empirical_scaling_factor)
        output_path = os.path.join(output_dir, "frame{:04d}_person{:02d}.png".format(frame_id, person_id))
        cv2.imwrite(output_path, frame_with_chicken_head)
