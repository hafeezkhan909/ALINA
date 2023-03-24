import cv2
import numpy as np
from final2 import normalize_color_features
from final3 import calc_histogram
from final4 import circular_threshold_pixel_discovery_and_traversal
import time
import os

start_time = time.time()
filename = '05640.jpg'
img = cv2.imread('/home/hafeez/Desktop/05640.jpg')
output_directory = '/home/hafeez/Desktop/i/'
final_img = img.copy()
# 42371
# 06449
# 06666
# 36578
# 41858

# Define the inverse trapezoidal region of interest (ROI) in the image
# bottom_width = 540
# top_width = 340
# height = 80

roi1 = np.array([[(540, 770), (490, 700),
                 (1210, 700), (1150, 770)]],
               dtype=np.int32)

'''
roi1 = np.array([[(540, 780), (490, 700),
                 (1600, 700), (1550, 780)]],
               dtype=np.int32)
'''
img1 = img.copy()

cv2.polylines(img1, roi1, True, (0, 0, 255), thickness=2)
cv2.imshow('Polyline', img1)
# print(roi.shape)
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
cv2.imwrite(os.path.join(output_directory, 'warped.jpg'), warped)
color_features = normalize_color_features(warped).reshape(-1, 3)

# Load your image
img1 = cv2.imread('/home/hafeez/Desktop/combined_features.jpg')
cv2.imshow('original', img1)
cv2.imwrite(os.path.join(output_directory, 'img1.jpg'), img1)
#cv2.waitKey()

# Convert to HSV color space
#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#cv2.imshow('HSV', hsv)

#cv2.waitKey()

# C
# Define the lower and upper ranges of yellow hue values
lower_yellow = np.array([0, 70, 150]) # lower hue value of yellow
upper_yellow = np.array([255, 255, 255]) # upper hue value of yellow

# T + C + G
# lower_yellow = np.array([40, 65, 0]) # lower hue value of yellow
# upper_yellow = np.array([120, 110, 120]) # upper hue value of yellow


# Create a binary mask
mask = cv2.inRange(img1, lower_yellow, upper_yellow)
cv2.imshow('Mask', mask)
cv2.imwrite(os.path.join(output_directory, 'mask.jpg'), mask)
print(mask.shape)

# Create mask
masked = np.ones(img1.shape[:2], dtype=np.uint8)
masked[:, 0:300] = 0

# Apply mask to image
masked_img = cv2.bitwise_and(mask, mask, mask=masked)

# Display masked image
cv2.imshow('Masked Image', masked_img)
cv2.imwrite(os.path.join(output_directory, 'masked_img.jpg'), masked_img)
print(masked_img.shape)
cv2.imwrite('/home/hafeez/Desktop/features.jpg', masked_img)
#cv2.waitKey()
# Apply the mask to the original image to extract yellow pixels
yellow_pixels = cv2.bitwise_and(warped, warped, mask=mask)
#cv2.imwrite('/home/hafeez/Desktop/features.jpg', yellow_pixels)

# Display the result
cv2.imshow('Yellow Pixels', yellow_pixels)
cv2.imwrite(os.path.join(output_directory, 'yellow_pixels.jpg'), yellow_pixels)
#cv2.waitKey(0)

from matplotlib import pyplot as plt
from statistics import mean

# Load the image
img2 = cv2.imread('/home/hafeez/Desktop/features.jpg', cv2.IMREAD_GRAYSCALE)

avg_pixel = calc_histogram(img2)

img3 = cv2.imread('/home/hafeez/Desktop/features.jpg', 0)
# Define the circular threshold
threshold = 20


# Define a visited array to keep track of the pixels that have already been processed
visited = np.zeros_like(img3)


# Define a list to store the coordinates of the white pixels that belong to the same line marking
line_marking_pixels = []




# Define a recursive function to find neighboring white pixels within the circular threshold
# Circular Threshold Pixel Exploration
# Circular Threshold Pixel Discovery and Traversal

#visited = set()
circular_threshold_pixel_discovery_and_traversal(img3, avg_pixel[0], avg_pixel[1], threshold, visited, line_marking_pixels)


# Call the flood_fill function with the starting point coordinates
#circular_threshold_pixel_discovery_and_traversal(avg_pixel[0], avg_pixel[1])


# Print the list of white pixels that belong to the same line marking
print(line_marking_pixels)


# Define the size of the black image to create
height, width = img3.shape[:2]


# Create a new black image
black_img = np.zeros((height, width), dtype=np.uint8)


# Plot the white pixels on the black image
for x, y in line_marking_pixels:
   cv2.circle(black_img, (x, y), 1, 255, -1)


# Display the black image with the white pixels plotted
cv2.imshow('Line Marking', black_img)
cv2.imwrite(os.path.join(output_directory, 'black_img.jpg'), black_img)
#cv2.waitKey()


# Compute the inverse perspective transform matrix
Minv = cv2.getPerspectiveTransform(dst, roi)


# Apply the inverse perspective transformation to the warped image
unwarped = cv2.warpPerspective(black_img, Minv, (img.shape[1], img.shape[0]))
cv2.imwrite(os.path.join(output_directory, 'unwarped.jpg'), unwarped)
cv2.imshow('Unwarped', unwarped)

# Assuming unwarped is a 2D array containing the image
line_pixels_only = np.where(unwarped > 0)
x_coords, y_coords = line_pixels_only[1], line_pixels_only[0]

# Stack the x and y coordinates horizontally
coords = np.column_stack((x_coords, y_coords))

# Save the coordinates to a text file
np.savetxt('/home/hafeez/Desktop/white_pixels.txt', coords, fmt='%6d')
#np.savetxt('/home/hafeez/Desktop/contour_ptsK2.txt', init_contour, fmt='%6d')



# Mark the white pixels in img2 with red
final_img[line_pixels_only] = (0, 0, 255)  # set pixel color to red
print('hi')
print(line_pixels_only)
print('hi')
print(len(line_pixels_only))


# Display the marked image
cv2.imshow('Marked Image', final_img)

cv2.imwrite(os.path.join(output_directory, 'final_img.jpg'), final_img)

end_time = time.time()


elapsed_time = end_time - start_time


print(f"Elapsed time: {elapsed_time: .2f} seconds")


cv2.waitKey()
cv2.destroyAllWindows()