import cv2
import numpy as np
from final2 import normalize_color_features
from final3 import calc_histogram
from final5 import circular_threshold_pixel_discovery_and_traversal
from datetime import datetime
import os

input_directory = '/home/hafeez/Desktop/5_vid'

# Get list of image filenames in ascending order
image_filenames = sorted([f for f in os.listdir(input_directory) if f.endswith('.jpg')])

output_directory1 = '/home/hafeez/Desktop/5_vid_annotations/'
output_directory2 = '/home/hafeez/Desktop/5_vid_textfiles/'

def active(img):

    final_img = img.copy()

    roi1 = np.array([[(540, 770), (490, 700),
                     (1210, 700), (1150, 770)]],
                   dtype=np.int32)

    img1 = img.copy()
    no_lines_img = img.copy()

    cv2.polylines(img1, roi1, True, (0, 0, 255), thickness=2)

    # Define the region of interest as a trapezoid
    roi = np.array([[(540, 770), (490, 700),
                    (1210, 700), (1150, 770)]],
                  dtype=np.float32)

    # Define the desired rectangular shape
    dst = np.array([[(50, 800), (50, 100),
                   (1200, 100), (1200, 800)]],
                 dtype=np.float32)

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(roi, dst)

    # Apply the perspective transformation to the image
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

    color_features = normalize_color_features(warped).reshape(-1, 3)

    # Load your image
    img1 = cv2.imread('/home/hafeez/Desktop/combined_features.jpg')

    # Define the lower and upper ranges of yellow hue values
    lower_yellow = np.array([0, 70, 150]) # lower hue value of yellow
    upper_yellow = np.array([255, 255, 255]) # upper hue value of yellow

    # Create a binary mask
    mask = cv2.inRange(img1, lower_yellow, upper_yellow)

    # Create mask
    masked = np.ones(img1.shape[:2], dtype=np.uint8)
    masked[:, 0:300] = 0

    # Apply mask to image
    masked_img = cv2.bitwise_and(mask, mask, mask=masked)

    img2 = masked_img.copy()
    img3 = masked_img.copy()

    # Apply the mask to the original image to extract yellow pixels
    yellow_pixels = cv2.bitwise_and(warped, warped, mask=mask)

    avg_pixel, peak_value = calc_histogram(img2)
    print(peak_value)
    if(peak_value > 75):
        # Define the circular threshold
        threshold = 20

        # Define a visited array to keep track of the pixels that have already been processed
        visited = np.zeros_like(img3)


        # Define a list to store the coordinates of the white pixels that belong to the same line marking
        line_marking_pixels = []

        visited = set()
        circular_threshold_pixel_discovery_and_traversal(img3, avg_pixel[0], avg_pixel[1], threshold, visited, line_marking_pixels)

        # Define the size of the black image to create
        height, width = img3.shape[:2]

        # Create a new black image
        black_img = np.zeros((height, width), dtype=np.uint8)

        # Plot the white pixels on the black image
        for x, y in line_marking_pixels:
           cv2.circle(black_img, (x, y), 1, 255, -1)

        # Compute the inverse perspective transform matrix
        Minv = cv2.getPerspectiveTransform(dst, roi)

        # Apply the inverse perspective transformation to the warped image
        unwarped = cv2.warpPerspective(black_img, Minv, (img.shape[1], img.shape[0]))

        line_pixels_only = np.where(unwarped > 0)
        x_coords, y_coords = line_pixels_only[1], line_pixels_only[0]

        # Stack the x and y coordinates horizontally
        coords = np.column_stack((x_coords, y_coords))

        image_basename = os.path.splitext(filename)[0]
        text_filename = image_basename + '.txt'
        print(text_filename)
        np.savetxt(output_directory2 + text_filename, coords, fmt='%6d')
        # Mark the white pixels in img2 with red
        final_img[line_pixels_only] = (0, 0, 255)  # set pixel color to red
        #print(line_pixels_only)

        # Display the marked image
        #cv2.imshow('Marked Image', final_img)
        print(filename)
        cv2.imwrite(os.path.join(output_directory1, filename), final_img)

        end_time, start_datetime, end_datetime, elapsed_time = time_calc()
        time_data = f"{filename}: Start time - {start_time}, End time - {end_time}, Elapsed time - {elapsed_time} seconds\n"
        f.write(time_data)

    else:
        # the image to be stored as it is and an empty text file
        image_basename = os.path.splitext(filename)[0]
        text_filename = image_basename + '.txt'
        print(text_filename)
        empty_arr = np.array([])
        np.savetxt(output_directory2 + text_filename, empty_arr)
        print(filename)
        cv2.imwrite(os.path.join(output_directory1, filename), no_lines_img)

        end_time, start_datetime, end_datetime, elapsed_time = time_calc()
        time_data = f"[No lines found in the image], {filename}: Start time - {start_time}, End time - {end_time}, Elapsed time - {elapsed_time} seconds\n"
        f.write(time_data)


def time_calc():
    end_time = datetime.now().strftime("%H:%M:%S")
    start_datetime = datetime.strptime(start_time, "%H:%M:%S")
    end_datetime = datetime.strptime(end_time, "%H:%M:%S")
    elapsed_time = (end_datetime - start_datetime).total_seconds()
    return end_time, start_datetime, end_datetime, elapsed_time


# Open the file for writing
with open('/home/hafeez/Desktop/5_vid_time_calc/processing_times2.txt', 'w') as f:
    for filename in image_filenames:
        if filename.endswith(".jpg"):
            start_time = datetime.now().strftime("%H:%M:%S")
            # Load the image
            img = cv2.imread(os.path.join(input_directory, filename))
            active(img)

cv2.destroyAllWindows()
