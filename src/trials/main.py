import cv2
import numpy as np
from sklearn.cluster import KMeans
from recursiveAlgorithm import recursive_algo
from automatedCanny import auto_canny
from extraction import normalize_color_features, normalize_texture_features, extract_road


file = "/home/hafeez/Desktop/cropped_images/00052.jpg"
img = cv2.imread(file)
cv2.imshow("original Image", img)
print(img.shape)

rgb2 = cv2.imread(file)
rgb3 = cv2.imread(file)
rgb4 = cv2.imread(file)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


edges = auto_canny(gray)
rows, cols = np.where(edges == 255)
print(rows)
print(cols)
coords = np.column_stack((cols, rows))
print(coords)

# Simplify the edges using the Ramer-Douglas-Peucker algorithm
threshold = 0.1  # adjust as needed
simplified_coords = np.array(recursive_algo(coords, threshold))
print(simplified_coords)

# Find lines in the simplified edges using the Hough transform
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)

# Draw the lines on the image
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Draw the simplified edges on the image
for coord in simplified_coords:
    pixel_color = rgb2[coord[1], coord[0]]
    cv2.circle(img, (coord[0], coord[1]), 1, (0, 0, 255), 2)

# Show the image
cv2.imshow("Image with lines", img)

y = lines[0][0][1]
cropped_image = rgb2[y:, :]
cv2.imshow("cropped image", cropped_image)
print(cropped_image.shape)

# Train KMeans model
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=500, tol=0.0001, verbose=0, random_state=None, copy_x=True)
color_features = normalize_color_features(rgb3).reshape(-1, 3)
texture_features = normalize_texture_features(rgb3).reshape(-1, 3)
multi_features = np.concatenate((color_features, texture_features), axis=-1)
kmeans.fit(multi_features)

# Extract the road from the image
road1 = extract_road(cropped_image, kmeans)
road2 = extract_road(rgb4, kmeans)

# Display the result
cv2.imshow('Result1', road1)
cv2.imshow('Result2', road2)
cv2.waitKey(0)
cv2.destroyAllWindows()