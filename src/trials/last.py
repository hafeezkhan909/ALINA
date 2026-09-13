import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import data
from skimage.feature import canny

def auto_canny(image, sigma=0.77):
    intensity = np.median(image)
    lower = int(max(0, (1.0 - sigma) * intensity))
    higher = int(max(255, (1.0 + sigma) * intensity))
    edge = cv2.Canny(img, lower, higher)
    cv2.imshow("Canny", edge)
    return edge

# Define the callback function for mouse events
def draw_contours(event, x, y, flags, param):
    global img, drawing, ix, iy, contour_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        contour_pts.append((x,y))
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(img, (ix, iy), (x, y), (134, 111, 205), 2)
            ix, iy = x, y
            contour_pts.append((x,y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (ix, iy), (x, y), (134, 111, 205), 2)
        contour_pts.append((x,y))

def active(img, init):
    # Create a binary mask from the contour
    mask = np.zeros_like(img)
    pts = np.array(init, dtype=np.int32)
    cv2.fillPoly(mask, [pts], (255, 255, 255))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Extract the region inside the contour
    region = cv2.bitwise_and(img, img, mask=mask)

    # Display the region inside the contour
    cv2.imshow('Region Inside Contour', region)
    cv2.waitKey()

    # Convert the region to grayscale
    region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    grrr = cv2.GaussianBlur(region_gray, (3, 3), 0)
    crrr = cv2.Canny(grrr, 40, 160)

    cv2.imshow('Canny of the Inside Region', crrr)
    cv2.waitKey()



# Load the image
img = cv2.imread('/home/hafeez/Desktop/keypoint_image.jpg')

# Create a window and bind the mouse events to the callback function
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_contours)

# Initialize some variables
drawing = False
contour_pts = []

while True:
    cv2.imshow('image', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Save the image
        cv2.imwrite('/home/hafeez/Desktop/image_with_contours.jpg', img)
        init_contour = np.array(contour_pts, np.int32)
        print(init_contour)
        img2 = cv2.imread('/home/hafeez/Desktop/ccc.jpg')
        active(img2, init_contour)
    elif key == 27:  # Press 'Esc' to exit
        break

cv2.destroyAllWindows()
