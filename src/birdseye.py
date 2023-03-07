import cv2
import numpy as np

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

def normalize_texture_features(image):
    filters = build_filters()
    texture_features = process(image, filters)
    texture_features = cv2.normalize(texture_features, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow("TNormalized", texture_features)
    return texture_features

def auto_canny(image, sigma=0.33):
    intensity = np.median(image)
    lower = int(max(0, (1.0 - sigma) * intensity))
    higher = int(max(255, (1.0 + sigma) * intensity))
    edge = cv2.Canny(image, lower, higher)
    return edge

img = cv2.imread('/home/hafeez/Desktop/36578.jpg')
# 42371
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
texture = normalize_texture_features(warped)
img_gray = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(img_gray, 30, 160)
cv2.imshow('Canny', canny)
# Compute the inverse perspective transform matrix
Minv = cv2.getPerspectiveTransform(dst, roi)

# Apply the inverse perspective transformation to the warped image
unwarped = cv2.warpPerspective(canny, Minv, (img.shape[1], img.shape[0]))

# texture = normalize_texture_features(warped)

# Display the original and warped images side by side
cv2.imshow('Original', img1)
cv2.imshow('Warped', warped)
cv2.imwrite('/home/hafeez/Desktop/warped.jpg', warped)
cv2.imshow('Unwarped', unwarped)
cv2.waitKey()
