import cv2
import numpy as np


# Load your image
img = cv2.imread('/home/hafeez/Desktop/combined_features.jpg')
warped = cv2.imread('/home/hafeez/Desktop/warped.jpg')
cv2.imshow('original', img)
cv2.waitKey()

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
mask = cv2.inRange(img, lower_yellow, upper_yellow)
cv2.imshow('Mask', mask)
print(mask.shape)

# Create mask
masked = np.ones(img.shape[:2], dtype=np.uint8)
masked[:, 0:300] = 0

# Apply mask to image
masked_img = cv2.bitwise_and(mask, mask, mask=masked)

# Display masked image
cv2.imshow('Masked Image', masked_img)

cv2.imwrite('/home/hafeez/Desktop/features.jpg', masked_img)
cv2.waitKey()
# Apply the mask to the original image to extract yellow pixels
yellow_pixels = cv2.bitwise_and(warped, warped, mask=mask)
#cv2.imwrite('/home/hafeez/Desktop/features.jpg', yellow_pixels)

# Display the result
cv2.imshow('Yellow Pixels', yellow_pixels)
cv2.waitKey(0)
cv2.destroyAllWindows()
