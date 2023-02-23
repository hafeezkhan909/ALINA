import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from scipy.interpolate import interp1d

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

# Define the callback function for mouse events
def draw_contours(event, x, y, flags, param):
    global img, drawing, ix, iy, contour_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        contour_pts.append((x, y))
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(img, (ix, iy), (x, y), (134, 111, 205), 2)
            ix, iy = x, y
            contour_pts.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (ix, iy), (x, y), (134, 111, 205), 2)
        contour_pts.append((x, y))

def simplify_con(final_img, contour_points):

    # assume contour_points is the variable containing the OpenCV format contour points
    point_list = []
    for point in contour_points:
        x, y = point[0]
        point_list.append((x, y))
        # print("This is point list")
        # print(point_list)
    point_list = np.array(point_list)





def active(img, init):
    # Create a copy of the image for display purposes
    img_display = np.copy(img)
    img2 = np.copy(img)
    img3 = np.copy(img)
    # Draw the initial contour on the image
    cv2.polylines(img_display, np.int32([init]), True, (0, 165, 255), thickness=2)
    # cv2.imshow("ddd", img_display)
    # Create a mask of zeros with the same shape as the image

    mask = np.zeros_like(img[:, :, 0])
    # Draw the initial contour on the mask with white color (255)
    cv2.fillPoly(mask, np.int32([init]), 255)

    # Apply Canny edge detection inside the contour region
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', img_gray)

    img_gray_masked = cv2.bitwise_and(img_gray, img_gray, mask=mask)
    cv2.imshow('Gray masked', img_gray_masked)

    img_gray_masked = cv2.GaussianBlur(img_gray_masked, (5, 5), 0)
    canny = cv2.Canny(img_gray_masked, 30, 160)
    cv2.imshow('Canny on the mask', canny)

    # Draw the initial contour on the canny image now
    cv2.polylines(canny, np.int32([init]), True, (0, 0, 0), thickness=2)
    cv2.imshow('pure', canny)


    rows, cols = np.where(canny == 255)
    print(rows)
    print(cols)
    coords = np.column_stack((cols, rows))
    print(coords)

    # Simplify the edges using the Ramer-Douglas-Peucker algorithm
    threshold = 0.1  # adjust as needed
    simplified_coords = np.array(recursive_algo(coords, threshold))
    print(simplified_coords)

    # Find lines in the simplified edges using the Hough transform
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)

    # Draw the lines on the image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw the simplified edges on the image
    for coord in simplified_coords:
        pixel_color = img2[coord[1], coord[0]]
        cv2.circle(img, (coord[0], coord[1]), 1, (0, 0, 255), 2)

    # Show the image
    cv2.imshow("Image with lines", img)
    cv2.waitKey()

    # Create a new list to store the final contour points
    final_contour = []

    # Add the points from the Hough transform lines to the final contour
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            final_contour.append([x1, y1])
            final_contour.append([x2, y2])

    # Add the simplified points to the final contour
    for coord in simplified_coords:
        final_contour.append([coord[0], coord[1]])

    # Convert the final contour points to a numpy array
    final_contour = np.array(final_contour).reshape((-1, 1, 2))
    print('finnanjknadk')
    print(final_contour)

    # Simplify the combined line using the Ramer-Douglas-Peucker algorithm
    final_line = cv2.approxPolyDP(final_contour, epsilon=0.001, closed=True)
    # Draw the approximated contour on the image
    cv2.polylines(img2, final_line, True, (0, 192, 255), thickness=2)
    # Show the image
    cv2.imshow("Final Contour", img2)
    cv2.waitKey()
    simplify_con(img3, final_contour)

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