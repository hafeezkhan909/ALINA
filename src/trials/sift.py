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

file = "/home/hafeez/Desktop/ccc.jpg"
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




gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
cv2.imshow("GRAY", gray)
# Create a SIFT object
sift = cv2.SIFT_create(nfeatures=500, nOctaveLayers=35, contrastThreshold=0.009, edgeThreshold=50, sigma=1.5)
# nfeatures: This parameter determines the maximum number of keypoints that will be detected in the image. By increasing the value of this parameter, you can increase the number of keypoints detected on the line markings,
# which can improve the chances of detecting thin, weak contrast line markings. However, increasing this value too much may also result in detecting keypoints in non-line marking regions, leading to false detections.

# nOctaveLayers: This parameter in the SIFT algorithm controls the number of layers per octave used in the construction of the scale space. The scale space is a pyramid of images in which each level is created by blurring the
# previous level with a Gaussian kernel and down-sampling the blurred image. The number of octave layers has a significant impact on the detection of scale-invariant keypoints in the image. Increasing the number of octave
# layers can help detect keypoints at smaller scales, but also increases the computational cost of the algorithm. Conversely, decreasing the number of octave layers reduces the computational cost, but may cause the algorithm
# to miss smaller-scale features. For detecting thin, weak contrast line markings, you may want to increase the number of octave layers to capture the features at different scales. However, this may increase the computational
# cost, so you may need to balance the number of layers with the available computational resources. You may also want to experiment with different values of the contrastThreshold and edgeThreshold parameters to optimize the
# detection of these features.

# contrastThreshold: This parameter sets the threshold for the contrast of the detected keypoints. By lowering this threshold, you can detect keypoints with lower contrast, which can be helpful for detecting weak contrast
# line markings. However, lowering this threshold too much can result in detecting too many keypoints in non-line marking regions, leading to false detections.

# edgeThreshold: This parameter sets the threshold for the edge response of the detected keypoints. By increasing this threshold, you can detect keypoints with higher edge responses, which can help to detect thin line markings.
# However, increasing this threshold too much can also result in detecting keypoints in non-line marking regions, leading to false detections.

# sigma: This parameter controls the scale of the detected keypoints. By decreasing this value, you can detect smaller keypoints, which can help to detect thin line markings. However, decreasing this value too much can also
# result in detecting keypoints in non-line marking regions, leading to false detections.

# sift = cv2.SIFT_create(nfeatures=150, nOctaveLayers=28, contrastThreshold=0.05, edgeThreshold=50, sigma=1.2)
# Detect the keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw the detected keypoints on the image
img_with_keypoints = cv2.drawKeypoints(cropped_image, keypoints, None)

# Display the image with keypoints
cv2.imshow("Keypoints", img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Filter the keypoints to retain only those that correspond to line markings
line_keypoints = []
for kp in keypoints:
    x, y = kp.pt
    if img[int(y), int(x), 0] < 50:  # Filter based on color (blue channel)
        line_keypoints.append(kp)

# Compute SIFT descriptors for the retained keypoints
line_descriptors = sift.compute(img, line_keypoints)[1]
