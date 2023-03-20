import cv2
import numpy as np
from final2 import normalize_color_features
from final3 import calc_histogram
from final4 import circular_threshold_pixel_discovery_and_traversal
import time
import os

input_directory = '/home/hafeez/Desktop/i'

# Get list of image filenames in ascending order
image_filenames = sorted([f for f in os.listdir(input_directory) if f.endswith('.jpg')])

output_directory = '/home/hafeez/Desktop/5_vid_images/'

start_time = time.time()

def active(img):

    final_img = img.copy()

    roi1 = np.array([[(540, 770), (490, 700),
                     (1210, 700), (1150, 770)]],
                   dtype=np.int32)

    img1 = img.copy()

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

    avg_pixel = calc_histogram(img2)

    # Define the circular threshold
    threshold = 30

    # Define a visited array to keep track of the pixels that have already been processed
    visited = np.zeros_like(img3)


    # Define a list to store the coordinates of the white pixels that belong to the same line marking
    line_marking_pixels = []

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

    # Find the white pixels in img1
    line_pixels_only = np.where(unwarped == 255)

    # Mark the white pixels in img2 with red
    final_img[line_pixels_only] = (0, 0, 255)  # set pixel color to red
    #print(line_pixels_only)

    # Display the marked image
    #cv2.imshow('Marked Image', final_img)
    print(filename)
    cv2.imwrite(os.path.join(output_directory, filename), final_img)


for filename in image_filenames:
    if filename.endswith(".jpg"):
        # Load the image
        img = cv2.imread(os.path.join(input_directory, filename))
        active(img)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time: .2f} seconds")

cv2.destroyAllWindows()
