import cv2
import os

# Function to save frames from video
def save_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Initialize frame count
    frame_count = 0

    while True:
        # Read a frame
        ret, frame = video.read()

        # If the frame was not retrieved, break the loop (end of video)
        if not ret:
            break

        # Create a filename with the format 00001.jpg, 00002.jpg, etc.
        filename = os.path.join(output_folder, f"{frame_count+1:05d}.jpg")

        # Save the frame as a JPEG file
        cv2.imwrite(filename, frame)

        # Increment frame count
        frame_count += 1

    # Release the video object
    video.release()
    print(f"Saved {frame_count} frames to {output_folder}")

# Example usage
video_path = '/zfsauton2/home/hshah2/projects/Grounded-SAM-2/demo_images/LaneChange.mp4'  # Replace with the path to your video file
output_folder = '/zfsauton2/home/hshah2/projects/Grounded-SAM-2/notebooks/videos/LaneChange'  # Replace with your desired output folder
save_frames(video_path, output_folder)
