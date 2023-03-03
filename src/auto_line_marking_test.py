import os
import numpy as np
import cv2
import time

start_time = time.time()

input_directory = '/home/hafeez/Desktop/K1/'

# Get list of image filenames in ascending order
image_filenames = sorted([f for f in os.listdir(input_directory) if f.endswith('.jpg')])

output_directory = '/home/hafeez/Desktop/K1_Auto_IMAGES'
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

def active(img, init, filename):

    if not active.has_been_called:
        print("This will be printed only once")
        # Create a copy of the image for display purposes
        img_display = np.copy(img)
        img2 = np.copy(img)

        # Draw the initial contour on the image
        cv2.polylines(img_display, np.int32([init]), True, (0, 165, 255), thickness=2)
        # cv2.imshow("ddd", img_display)
        # Create a mask of zeros with the same shape as the image

        mask = np.zeros_like(img[:, :, 0])
        # Draw the initial contour on the mask with white color (255)
        cv2.fillPoly(mask, np.int32([init]), 255)

        # Apply Canny edge detection inside the contour region
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Gray', img_gray)

        img_gray_masked = cv2.bitwise_and(img_gray, img_gray, mask=mask)
        # cv2.imshow('Gray masked', img_gray_masked)

        img_gray_masked = cv2.GaussianBlur(img_gray_masked, (5, 5), 0)
        canny = cv2.Canny(img_gray_masked, 30, 160)
        # cv2.imshow('Canny on the mask', canny)

        # Draw the initial contour on the canny image now
        cv2.polylines(canny, np.int32([init]), True, (0, 0, 0), thickness=2)
        # cv2.imshow('pure', canny)


        rows, cols = np.where(canny == 255)
        print(rows)
        print(cols)
        coords = np.column_stack((cols, rows))
        print(coords)

        # Simplify the edges using the Ramer-Douglas-Peucker algorithm
        threshold = 0.9 # adjust as needed
        simplified_coords = np.array(recursive_algo(coords, threshold))
        print(simplified_coords)

        # Find lines in the simplified edges using the Hough transform
        lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 4, minLineLength=2.5, maxLineGap=0.5)

        # Draw the lines on the image
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw the simplified edges on the image
        for coord in simplified_coords:
            pixel_color = img2[coord[1], coord[0]]
            cv2.circle(img, (coord[0], coord[1]), 1, (0, 0, 255), 2)

        # Create a new list to store the final points of the line markings
        final_contour = []

        # Add the points from the Hough transform lines to the final contour
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Bresenham's line algorithm is being used to generate all the pixels on the line
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                sx = 1 if x1 < x2 else -1
                sy = 1 if y1 < y2 else -1
                err = dx - dy
                while x1 != x2 or y1 != y2:
                    final_contour.append([x1, y1])
                    e2 = 2 * err
                    if e2 > -dy:
                        err -= dy
                        x1 += sx
                    if e2 < dx:
                        err += dx
                        y1 += sy
                final_contour.append([x2, y2])

        # Add the simplified points to the final contour
        for coord in simplified_coords:
            final_contour.append([coord[0], coord[1]])

        image_basename = os.path.splitext(filename)[0]
        text_filename = image_basename + '.txt'
        print(text_filename)
        np.savetxt('/home/hafeez/Desktop/K1_Auto_TEXTFILES/' + text_filename, final_contour, fmt='%6d')

        # Show the image

        cv2.imshow("Image with lines", img)
        cv2.waitKey()
        # cv2.imwrite(os.path.join(output_directory, filename), img)
        active.has_been_called = True

    # Running in loop after 1st iteration
    img_display = np.copy(img)
    img2 = np.copy(img)

    # Draw the initial contour on the image
    cv2.polylines(img_display, np.int32([init]), True, (0, 165, 255), thickness=2)
    # cv2.imshow("ddd", img_display)
    # Create a mask of zeros with the same shape as the image

    mask = np.zeros_like(img[:, :, 0])
    # Draw the initial contour on the mask with white color (255)
    cv2.fillPoly(mask, np.int32([init]), 255)

    # Apply Canny edge detection inside the contour region
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Gray', img_gray)

    img_gray_masked = cv2.bitwise_and(img_gray, img_gray, mask=mask)
    # cv2.imshow('Gray masked', img_gray_masked)

    img_gray_masked = cv2.GaussianBlur(img_gray_masked, (5, 5), 0)
    canny = cv2.Canny(img_gray_masked, 30, 160)
    # cv2.imshow('Canny on the mask', canny)

    # Draw the initial contour on the canny image now
    cv2.polylines(canny, np.int32([init]), True, (0, 0, 0), thickness=2)
    # cv2.imshow('pure', canny)

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
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 4, minLineLength=2.5, maxLineGap=0.5)

    # Draw the lines on the image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw the simplified edges on the image
    for coord in simplified_coords:
        pixel_color = img2[coord[1], coord[0]]
        cv2.circle(img, (coord[0], coord[1]), 1, (0, 0, 255), 2)

    # Create a new list to store the final points of the line markings
    final_contour = []

    # Add the points from the Hough transform lines to the final contour
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Bresenham's line algorithm is being used to generate all the pixels on the line
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy
            while x1 != x2 or y1 != y2:
                final_contour.append([x1, y1])
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x1 += sx
                if e2 < dx:
                    err += dx
                    y1 += sy
            final_contour.append([x2, y2])

    # Add the simplified points to the final contour
    for coord in simplified_coords:
        final_contour.append([coord[0], coord[1]])

    image_basename = os.path.splitext(filename)[0]
    text_filename = image_basename + '.txt'
    np.savetxt('/home/hafeez/Desktop/K1_Auto_TEXTFILES/' + text_filename, final_contour, fmt='%6d')

    cv2.imwrite(os.path.join(output_directory, filename), img)
    # cv2.waitKey()

active.has_been_called = False
def func_one():
    # Load the contour points from file
    contour_pts = np.loadtxt('/home/hafeez/Desktop/contour_pts.txt', dtype=np.int64)

    # Iterate through all the images in the directory

    for filename in image_filenames:
        if filename.endswith(".jpg"):
            # Load the image
            img = cv2.imread(os.path.join(input_directory, filename))

            # Draw the contour
            # cv2.polylines(img, [contour_pts], True, (0, 255, 0), 2)
            active(img, contour_pts, filename)
            # cv2.imwrite(os.path.join(directory, "contoured_" + filename), img)
            # Call the lines function
            # lines(img)
    cv2.destroyAllWindows()

# Load the initial image
img = cv2.imread('/home/hafeez/Desktop/K1/08001.jpg')
# 42371
# 36578

# Define the inverse trapezoidal region of interest (ROI) in the image
# bottom_width = 540
# top_width = 340
# height = 80

roi = np.array([[(540, 770), (490, 700),
                 (1210, 700), (1150, 770)]],
               dtype=np.int32)

# Draw the ROI on the image
#cv2.polylines(img, roi, True, (0, 0, 255), thickness=2)

mask = np.zeros_like(img[:, :, 0])
cv2.fillPoly(mask, roi, 255)
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(contours)
roi_img = cv2.bitwise_and(img, img, mask=mask)
init_contours = np.squeeze(contours)
print('hi')
print(init_contours)
np.savetxt('/home/hafeez/Desktop/contour_pts.txt', init_contours, fmt='%6d')
cv2.imshow('Image', roi_img)
cv2.waitKey()
func_one()
# active(img1, init_contours, file)
cv2.destroyAllWindows()
