import cv2
import numpy as np

# Load image in grayscale

def auto_canny(image, sigma=0.33):
    intensity = np.median(image)
    lower = int(max(0, (1.0 - sigma) * intensity))
    higher = int(max(255, (1.0 + sigma) * intensity))
    edge = cv2.Canny(image, lower, higher)
    return edge

img = cv2.imread('/home/hafeez/Desktop/vidd/40000.jpg')
'''
test images
02796
05622
'''
cv2.imwrite('/home/hafeez/Desktop/i/original_image.jpg', img)
cv2.imshow('Original Image', img)
cv2.waitKey()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image',gray)
# Apply Gaussian blur to reduce noise
img_blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges using Canny algorithm
# edges = cv2.Canny(img_blur, 40, 154)
edges = auto_canny(img_blur)

# lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=2.5, maxLineGap=5.5)
#
#         # Draw the lines on the image
# if lines is not None:
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Load x, y coordinates from text file
with open('/home/hafeez/Desktop/vidd_textfiles/40000.txt') as f:
    lines = f.readlines()

# Convert coordinates to numpy array
points = np.zeros((len(lines), 2), dtype=np.int32)
print(len(points))
for i, line in enumerate(lines):
    x, y = line.strip().split()
    points[i] = [int(x), int(y)]

img1 = img.copy()

edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
# Draw points on edge detection image
for point in points:
    # print(point)
    cv2.circle(img1, tuple(point), 2, (0, 0, 255), -1)
    cv2.circle(edges_color, tuple(point), 2, (0, 0, 255), -1)


# Display the results
cv2.imshow('Canny Edges', edges)
cv2.imshow('Super-imposed on canny image', edges_color)
# cv2.imshow('Hough Lines', img)
# cv2.imshow('Super-imposing the marked lines on Hough Lines', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
