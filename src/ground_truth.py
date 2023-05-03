import cv2
import numpy as np
import os

output_directory1 = "C:\\Users\\assist-lab\\Desktop\\canny_images2\\"
output_directory2 = "C:\\Users\\assist-lab\\Desktop\\canny_textfiles2\\"
# Define the callback function for mouse events
def draw_contours(event, x, y, flags, param):
    global img, drawing, ix, iy, contour_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        contour_pts.append((x, y))
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(img, (ix, iy), (x, y), (0, 255, 255), 2)
            ix, iy = x, y
            contour_pts.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (ix, iy), (x, y), (0, 255, 255), 2)
        contour_pts.append((x, y))


def active(img, init):
    # Create a copy of the image for display purposes
    img_display = np.copy(img)
    img2 = np.copy(img)
    img3 = np.copy(img)
    img4 = np.copy(img)

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
    canny = cv2.Canny(img_gray_masked, 30, 150)
    cv2.imshow('Canny on the mask', canny)
    cv2.waitKey()

    # Draw the initial contour on the canny image now
    cv2.polylines(canny, np.int32([init]), True, (0, 0, 0), thickness=2)
    cv2.imshow('pure', canny)
    cv2.waitKey()

    line_pixels_only = np.where(canny > 0)
    x_coords, y_coords = line_pixels_only[1], line_pixels_only[0]

    # Stack the x and y coordinates horizontally
    coords = np.column_stack((x_coords, y_coords))

    image_basename = os.path.splitext(filename)[0]
    text_filename = image_basename + '.txt'
    print(text_filename)
    cv2.imwrite(os.path.join(output_directory1, image_basename + '_canny.jpg'), canny)
    np.savetxt(output_directory2 + text_filename, coords, fmt='%6d')
    print('saved !!!')


# Load the image
image = 'C:\\Users\\assist-lab\\Desktop\\output\\2\\04071.jpg'
filename = '04071.jpg'
img = cv2.imread(image)
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
        # cv2.imwrite('/home/hafeez/Desktop/image_with_contours.jpg', img)
        init_contour = np.array(contour_pts, np.int32)
        print(init_contour)
        print('this is the length ', len(init_contour))
        img2 = cv2.imread(image)
        active(img2, init_contour)
    elif key == 27:  # Press 'Esc' to exit
        break

cv2.destroyAllWindows()