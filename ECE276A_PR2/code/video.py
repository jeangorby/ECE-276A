import cv2
import os

def create_video_from_images(image_folder, output_video, fps=30):
    # Get list of image files (adjust extensions as needed)
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort the images by filename (modify if you need a different sort order)
    from natsort import natsorted
    image_files = natsorted([f for f in os.listdir(image_folder)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        print("No images found in the folder:", image_folder)
        return

    # Read the first image to obtain the frame size (width, height)
    first_frame = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, channels = first_frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID' or other codecs
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Loop over sorted images and write to video
    for file_name in image_files:
        file_path = os.path.join(image_folder, file_name)
        frame = cv2.imread(file_path)
        if frame is None:
            print("Warning: Skipping file", file_path)
            continue
        # Optionally, resize or process the frame here if needed
        video_writer.write(frame)

    video_writer.release()
    print("Video saved as:", output_video)

if __name__ == "__main__":
    # Specify your folder containing the images and the desired output video filename
    image_folder = "../data/dataRGBD/RGB20"  # e.g., "./images"
    output_video = "trajectory_video_20.mp4"
    fps = 30  # Frames per second for the output video

    create_video_from_images(image_folder, output_video, fps)
