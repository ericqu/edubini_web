import cv2
import numpy as np
import argparse

def average_frames(video_path, output_path, sharpen=True):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Read the first frame to set up our 32-bit float accumulator array
    ret, frame = cap.read()
    if not ret:
        print("Error: Video contains no frames.")
        return
        
    print(f"Starting frame averaging for {video_path}...")
    
    # Initialize the accumulator and frame counter
    accumulator = np.float32(frame)
    frame_count = 1

    # Loop through all remaining frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Add the current frame's values to our accumulator
        cv2.accumulate(frame, accumulator)
        frame_count += 1

    cap.release()
    print(f"Successfully processed {frame_count} frames.")

    # Divide by total frames to get the true average
    average_float = accumulator / frame_count
    
    # Convert back to standard 8-bit image pixels (0-255)
    averaged_image = np.uint8(average_float)
    
    if sharpen:
        print("Applying convolution matrix sharpening...")
        # Define your custom 3x3 sharpening kernel
        kernel = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]], dtype=np.float32)

        # The '-1' tells cv2 to keep the output the same depth as the input
        output_image = cv2.filter2D(averaged_image, -1, kernel)
    else:
        output_image = averaged_image

    # Save the output image
    cv2.imwrite(output_path, output_image)
    print(f"Saved {'sharpened' if sharpen else 'unsharpened'} image to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Average all frames of a video into one image.")
    parser.add_argument("input_video", help="Input video file (e.g. .mov)")
    parser.add_argument("output_image", help="Output image file (e.g. .jpg)")
    parser.add_argument("--sharpen", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply sharpening (default: on; use --no-sharpen to disable)")
    args = parser.parse_args()
    average_frames(args.input_video, args.output_image, sharpen=args.sharpen)