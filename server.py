# import cv2
# import numpy as np

# # Load the video
# video_path = "30.mp4"
# cap = cv2.VideoCapture(video_path)

# # Get video properties
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames in input video

# # Define output video at 120 FPS
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output_120fps.mp4', fourcc, 120, (width, height))

# # Read the first frame
# ret, prev_frame = cap.read()
# current_frame = 1  # Start frame index

# while ret:
#     ret, next_frame = cap.read()
#     if not ret:
#         break

#     # First interpolated frame (pixel-wise average between previous and next frame)
#     middle_frame_1 = ((prev_frame.astype(np.float32) + next_frame.astype(np.float32)) / 2).astype(np.uint8)

#     # Second interpolated frame (pixel-wise average between previous frame and first interpolated frame)
#     middle_frame_2 = ((prev_frame.astype(np.float32) + middle_frame_1.astype(np.float32)) / 2).astype(np.uint8)

#     # Write original and interpolated frames
#     out.write(prev_frame)
#     out.write(middle_frame_1)  # First interpolated frame
#     out.write(middle_frame_2)  # Second interpolated frame
#     out.write(next_frame)      # Original frame B

#     prev_frame = next_frame  # Move to next frame

#     # Calculate progress percentage
#     progress = (current_frame / total_frames) * 100
#     print(f"Progress: {progress:.2f}% ({current_frame}/{total_frames} frames processed)")

#     current_frame += 1

# # Release resources
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# print("✅ Video processing completed!")


# # Color EQ code below
# import cv2
# import numpy as np

# # Load the video
# video_path = "30.mp4"
# cap = cv2.VideoCapture(video_path)

# # Get video properties
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames in input video

# # Define output video at 120 FPS
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output_120fps_equalized.mp4', fourcc, 120, (width, height))

# # CLAHE for color equalization (contrast enhancement)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# # Function to equalize colors
# def equalize_colors(frame):
#     # Convert to LAB color space
#     lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    
#     # Split the LAB image into L, A, and B channels
#     l, a, b = cv2.split(lab)
    
#     # Apply CLAHE to the L channel (luminance)
#     l = clahe.apply(l)
    
#     # Merge the channels back
#     lab = cv2.merge((l, a, b))
    
#     # Convert back to BGR color space
#     equalized_frame = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
#     return equalized_frame

# # Read the first frame
# ret, prev_frame = cap.read()
# current_frame = 1  # Start frame index

# while ret:
#     ret, next_frame = cap.read()
#     if not ret:
#         break

#     # First interpolated frame (pixel-wise average between previous and next frame)
#     middle_frame_1 = ((prev_frame.astype(np.float32) + next_frame.astype(np.float32)) / 2).astype(np.uint8)

#     # Second interpolated frame (pixel-wise average between previous frame and first interpolated frame)
#     middle_frame_2 = ((prev_frame.astype(np.float32) + middle_frame_1.astype(np.float32)) / 2).astype(np.uint8)

#     # Equalize colors for all frames (original and interpolated)
#     prev_frame = equalize_colors(prev_frame)
#     middle_frame_1 = equalize_colors(middle_frame_1)
#     middle_frame_2 = equalize_colors(middle_frame_2)
#     next_frame = equalize_colors(next_frame)

#     # Write original and interpolated frames with color equalization
#     out.write(prev_frame)
#     out.write(middle_frame_1)  # First interpolated frame
#     out.write(middle_frame_2)  # Second interpolated frame
#     out.write(next_frame)      # Original frame B

#     prev_frame = next_frame  # Move to next frame

#     # Calculate progress percentage
#     progress = (current_frame / total_frames) * 100
#     print(f"Progress: {progress:.2f}% ({current_frame}/{total_frames} frames processed)")

#     current_frame += 1

# # Release resources
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# print("✅ Video processing completed!")

#WeightedAverage

import cv2
import numpy as np

# Load the video
video_path = "output_30fps.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames in input video

# Define output video at 120 FPS
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_120fps_weighted_no_eq.mp4', fourcc, fps*4, (width, height))

# Function for weighted averaging (without color equalization)
def weighted_interpolation(prev_frame, next_frame, weight_left=0.3, weight_right=0.7):
    """Perform weighted averaging between two frames"""
    # First interpolated frame with weighted averaging (previous and next frame)
    middle_frame_1 = (prev_frame.astype(np.float32) * weight_left + next_frame.astype(np.float32) * weight_right).astype(np.uint8)

    # Second interpolated frame with weighted averaging (previous frame and first interpolated frame)
    middle_frame_2 = (prev_frame.astype(np.float32) * weight_left + middle_frame_1.astype(np.float32) * weight_right).astype(np.uint8)

    return middle_frame_1, middle_frame_2

# Read the first frame
ret, prev_frame = cap.read()
current_frame = 1  # Start frame index

while ret:
    ret, next_frame = cap.read()
    if not ret:
        break

    # Define the weights for interpolation
    weight_left = 0.3  # Weight for the previous frame
    weight_right = 0.7  # Weight for the next frame
    
    # Perform weighted interpolation
    middle_frame_1, middle_frame_2 = weighted_interpolation(prev_frame, next_frame, weight_left, weight_right)

    # Write original and interpolated frames
    out.write(prev_frame)
    out.write(middle_frame_1)  # First interpolated frame
    out.write(middle_frame_2)  # Second interpolated frame
    out.write(next_frame)      # Original frame B

    prev_frame = next_frame  # Move to next frame

    # Calculate progress percentage
    progress = (current_frame / total_frames) * 100
    print(f"Progress: {progress:.2f}% ({current_frame}/{total_frames} frames processed)")

    current_frame += 1

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Video processing completed!")





# # turn into 30fps video
# import cv2

# # Input and output video paths
# input_video = "30.mp4"  # Change this to your input video
# output_video = "output_30fps.mp4"

# # Load the video
# cap = cv2.VideoCapture(input_video)

# # Get original video properties
# original_fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# # Define output video writer (set FPS to 30)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# target_fps = 10
# out = cv2.VideoWriter(output_video, fourcc, target_fps, (width, height))

# # Frame selection logic
# frame_interval = original_fps / target_fps  # Determines how to sample frames
# frame_idx = 0  # Keeps track of frame index
# current_frame = 1

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Write frame if it's within the interval
#     if frame_idx % round(frame_interval) == 0:
#         out.write(frame)

#     frame_idx += 1

#     # Display progress
#     progress = (current_frame / total_frames) * 100
#     print(f"Progress: {progress:.2f}% ({current_frame}/{total_frames} frames processed)")
#     current_frame += 1

# # Release resources
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# print("✅ Video conversion to 30 FPS completed!")
