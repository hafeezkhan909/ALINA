# -*- coding: utf-8 -*-

import cv2
import time
import os

def main():
    pass
def rotate_frames(input_loc, output_loc):
    """Function to rotate frames by 180 degrees
        and save them in an output directory.
        Args:
            input_loc: Input frame directory.
            output_loc: Output directory to save the frames.
        Returns:
            None
        """
    count=0
    time_start = time.time()
    file_count = len(os.listdir(input_loc))
    print("Total size is %d\n" %file_count)
    for images in os.listdir(input_loc):
        src = cv2.imread(input_loc + images)
        image = cv2.rotate(src, cv2.ROTATE_180)
        print("Rotating Image at "+ images)
        cv2.imwrite(output_loc + images, image)
        count = count +1
    time_end = time.time()
    print("Done rotating the frames.\n%d frames rotated" % count)
    print("It took %d seconds for rotation." % (time_end - time_start))

def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    mal_count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            mal_count = mal_count +1
            continue
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        print("Frame %d being created " %count)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("There were %d Malformed Frames\n" %mal_count)
            print ("It took %d seconds for conversion." % (time_end-time_start))
            break



if __name__ == "__main__":
    rotate_list = [2]
    for i in rotate_list:
        input_loc = 'C:\\Users\\assist-lab\\Desktop\\input\\'+ str(i) + '.mp4'
        output_loc = 'C:\\Users\\assist-lab\\Desktop\\output\\' + str(i) + '\\'
        video_to_frames(input_loc, output_loc)
        # rotate_frames(input_loc, output_loc)