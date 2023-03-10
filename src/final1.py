import cv2
import numpy as np
from final2 import normalize_color_features
from final3 import calc_histogram
import time

start_time = time.time()

img = cv2.imread('/home/hafeez/Desktop/42371.jpg')
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

img1 = img.copy()

cv2.polylines(img1, roi1, True, (0, 0, 255), thickness=2)

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

color_features = normalize_color_features(warped).reshape(-1, 3)

# Load your image
img1 = cv2.imread('/home/hafeez/Desktop/combined_features.jpg')
cv2.imshow('original', img1)
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
print(mask.shape)

# Create mask
masked = np.ones(img1.shape[:2], dtype=np.uint8)
masked[:, 0:300] = 0

# Apply mask to image
masked_img = cv2.bitwise_and(mask, mask, mask=masked)

# Display masked image
cv2.imshow('Masked Image', masked_img)

cv2.imwrite('/home/hafeez/Desktop/features.jpg', masked_img)
#cv2.waitKey()
# Apply the mask to the original image to extract yellow pixels
yellow_pixels = cv2.bitwise_and(warped, warped, mask=mask)
#cv2.imwrite('/home/hafeez/Desktop/features.jpg', yellow_pixels)

# Display the result
cv2.imshow('Yellow Pixels', yellow_pixels)
#cv2.waitKey(0)

from matplotlib import pyplot as plt
from statistics import mean

# Load the image
img2 = cv2.imread('/home/hafeez/Desktop/features.jpg', cv2.IMREAD_GRAYSCALE)

avg_pixel = calc_histogram(img2)

# Compute the inverse perspective transform matrix
Minv = cv2.getPerspectiveTransform(dst, roi)

# Apply the inverse perspective transformation to the warped image
unwarped = cv2.warpPerspective(img2, Minv, (img.shape[1], img.shape[0]))
#cv2.imshow('Unwarped', unwarped)

# Find the white pixels in img1
line_pixels_only = np.where(unwarped == 255)

# Mark the white pixels in img2 with red
final_img[line_pixels_only] = (0, 0, 255)  # set pixel color to red
print(line_pixels_only)

# Display the marked image
cv2.imshow('Marked Image', final_img)

end_time = time.time()

elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time: .2f} seconds")

cv2.waitKey()
cv2.destroyAllWindows()