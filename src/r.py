import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
def recursive_algo(points, threshold):

    dmax = 0
    index = 0
    for i in range(1, len(points) - 1):
        d = dist_to_segment(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d

    if dmax > threshold:
        rec_results1 = recursive_algo(points[:index+1], threshold)
        rec_results2 = recursive_algo(points[index:], threshold)

        return rec_results1[:-1] + rec_results2
    else:
        return [points[0], points[-1]]

def dist_to_segment(p, v, w):
    """
    Distance between a point p and a line segment defined by points v and w.
    """
    l = dist_sq(v, w)
    if l == 0:
        return dist_sq(p, v)
    t = max(0, min(1, np.dot(p - v, w - v) / l))
    projection = v + t * (w - v)
    return dist_sq(p, projection)

def dist_sq(p1, p2):
    """
    Squared Euclidean distance between two points p1 and p2.
    """
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def auto_canny(image, sigma=0.33):
    intensity = np.median(image)
    lower = int(max(0, (1.0 - sigma) * intensity))
    higher = int(max(255, (1.0 + sigma) * intensity))
    edge = cv2.Canny(img, lower, higher)
    return edge

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def normalize_color_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX)
    s = cv2.normalize(s, None, 0, 255, cv2.NORM_MINMAX)
    v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    color_features = np.stack((h, s, v), axis=-1)
    cv2.imshow("CNormalized", color_features)
    return color_features

def normalize_texture_features(image):
    filters = build_filters()
    texture_features = process(image, filters)
    texture_features = cv2.normalize(texture_features, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow("TNormalized", texture_features)
    return texture_features

def gradient_normalization(image):
    gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
    cv2.imshow("GjfjfjNormalized", gradient_magnitude)
    return gradient_magnitude

def extract_road(image, kmeans_model):
    color_features = normalize_color_features(image)
    texture_features = normalize_texture_features(image)
    gradient_features = gradient_normalization(image)
    multi_features = np.concatenate((color_features, texture_features, gradient_features), axis=-1)
    multi_features = multi_features.reshape(-1, 9)
    print(multi_features)
    labels = kmeans_model.predict(multi_features)
    labels = labels.reshape(*image.shape[:2])

    color_img = np.zeros_like(image)
    n_clusters = kmeans_model.n_clusters
    colors = np.random.randint(0, 255, size=(n_clusters, 3))
    for k in range(n_clusters):
        cluster_mask = (labels == k)
        color_img[cluster_mask] = colors[k]
    return color_img

file = "/home/hafeez/Desktop/all_images/00007.jpg"
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

just = []
# Draw the simplified edges on the image
for coord in simplified_coords:
    pixel_color = rgb2[coord[1], coord[0]]
    cv2.circle(img, (coord[0], coord[1]), 1, (0, 0, 255), 2)
    just.append([coord[0], coord[1]])
print("The number of RDP points")
print(len(just))

# Show the image
cv2.imshow("Image with lines", img)

y = lines[0][0][1]
cropped_image = rgb2[y:, :]
cv2.imshow("cropped image", cropped_image)
print(cropped_image.shape)

# Train KMeans model
kmeans = KMeans(n_clusters=9, init='k-means++', n_init=10, max_iter=5000, tol=0.1, verbose=0, random_state=None, copy_x=True)
color_features = normalize_color_features(rgb3).reshape(-1, 3)
texture_features = normalize_texture_features(rgb3).reshape(-1, 3)
gradient_features = gradient_normalization(rgb3).reshape(-1, 3)
multi_features = np.concatenate((color_features, texture_features, gradient_features), axis=-1)
cv2.imshow('MMMMMMM', multi_features)
kmeans.fit(multi_features)

# Extract the road from the image
road1 = extract_road(cropped_image, kmeans)
road2 = extract_road(rgb4, kmeans)
#output_dir = "/home/hafeez/Desktop"
# Display the result
cv2.imshow('Result1', road1)
#cv2.imwrite(os.path.join(output_dir, 'MMM4.jpg'), road2)
cv2.imshow('Result2', road2)
cv2.waitKey(0)
cv2.destroyAllWindows()