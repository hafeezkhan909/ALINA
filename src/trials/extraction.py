import cv2
import numpy as np
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

def extract_road(image, kmeans_model):
    color_features = normalize_color_features(image)
    texture_features = normalize_texture_features(image)
    multi_features = np.concatenate((color_features, texture_features), axis=-1)
    multi_features = multi_features.reshape(-1, 6)
    labels = kmeans_model.predict(multi_features)
    labels = labels.reshape(*image.shape[:2])

    color_img = np.zeros_like(image)
    n_clusters = kmeans_model.n_clusters
    colors = np.random.randint(0, 255, size=(n_clusters, 3))
    for k in range(n_clusters):
        cluster_mask = (labels == k)
        color_img[cluster_mask] = colors[k]
    return color_img