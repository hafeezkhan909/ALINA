import cv2
import numpy as np
from matplotlib import pyplot as plt
from statistics import mean

# Load image
img = cv2.imread('/home/hafeez/Desktop/features.jpg', cv2.IMREAD_GRAYSCALE)

# Convert image to binary
binary_img = np.where(img == 255, 1, 0)

# Sum binary image along vertical axis
projection = np.sum(binary_img, axis=0)

# Plot projection of white pixels
plt.plot(projection)
plt.xlabel('X-coordinate')
plt.ylabel('Number of white pixels')
plt.show()

# Find peak value of projection
peak_value = np.max(projection)
peak_index = np.argmax(projection)

# Get x, y coordinates of white pixels in peak area
peak_area = binary_img[:, peak_index]
white_pixels = []
for i, row in enumerate(peak_area):
    if row == 1:
        # x = peak_index
        y = i
        white_pixels.append(y)

list_avg = int(mean(white_pixels))
avg_pixel = [peak_index, list_avg]

print(avg_pixel)
print('Peak value:', peak_value)
print('Peak index:', peak_index)
print('White pixel coordinates:', white_pixels)