import cv2
import numpy as np

POINT_COLOR = (0, 0, 255)
LINE_COLOR = (0, 255, 0)
LINE_THICKNESS = 2

def select_roi(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Create an empty image for drawing the ROI
    roi_img = np.copy(img)

    # Variables for tracking the line position
    start_point = None
    end_point = None
    roi_points = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal roi_points, roi_img, start_point, end_point

        if event == cv2.EVENT_LBUTTONDOWN:
            if start_point is None:
                start_point = (x, y)
            else:
                end_point = (x, y)
                roi_points.append(start_point)
                roi_points.append(end_point)
                #print(roi_points)
                #print(len(roi_points))
                start_point = end_point

        roi_img = np.copy(img)
        if start_point is not None:
            if len(roi_points) == 2:
                cv2.polylines(roi_img, [np.array(roi_points)], False, POINT_COLOR, LINE_THICKNESS)
                cv2.line(roi_img, (start_point[0], start_point[1]), (x, start_point[1]), LINE_COLOR, LINE_THICKNESS)
            # elif len(roi_points) == 6:
            #     cv2.polylines(roi_img, [np.array(roi_points)], False, POINT_COLOR, LINE_THICKNESS)
            #     cv2.line(roi_img, (start_point[0], start_point[1]), (x, start_point[1]), LINE_COLOR, LINE_THICKNESS)
            else:
                cv2.polylines(roi_img, [np.array(roi_points)], False, POINT_COLOR, LINE_THICKNESS)
                cv2.line(roi_img, start_point, (x, y), LINE_COLOR, LINE_THICKNESS)

        cv2.imshow('Polyline', roi_img)

    roi_points = []

    cv2.namedWindow('Polyline')
    cv2.setMouseCallback('Polyline', mouse_callback)
    cv2.imshow('Polyline', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    roi = np.array([roi_points], dtype=np.int32)
    #print(roi)
    #print(roi[0][1][1])
    #cv2.polylines(img, roi, True, POINT_COLOR, LINE_THICKNESS)
    #cv2.imshow('Polyline', img)
    #cv2.waitKey()

    return roi

# Usage:
# image_path = "C:\\Users\\assist-lab\\Desktop\\output\\3\\15134.jpg"
# roi_points = select_roi(image_path)
# print(roi_points)
