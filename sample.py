import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow 
video_capture = cv2.VideoCapture('1.mp4')
frame_number = 0
frame_skip = 5 
rotate_angle = 20
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)
rightmost_positions = {}
green_dot_positions_x = []
green_dot_positions_y = []
time_points = []
roi_y_start = 300 
roi_y_end = 450  
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    if frame_number == 0:
        rows, cols, _ = frame.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotate_angle, 1)
        frame = cv2.warpAffine(frame, M, (cols, rows))

    if frame_number % frame_skip == 0:
        roi = frame[roi_y_start:roi_y_end, :]
        fg_mask = bg_subtractor.apply(roi)
        edges = cv2.Canny(fg_mask, threshold1=50, threshold2=70)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        for i, row in enumerate(edges):
            if np.any(row > 0):
                rightmost_edge = np.max(np.where(row > 0))
                cv2.circle(roi, (rightmost_edge, i), 2, (255, 0, 0), -1)
                rightmost_positions[i] = rightmost_edge
            else:
                rightmost_positions[i] = -1  
        blue_edge_points = np.array([(v, k) for k, v in rightmost_positions.items() if v != -1])
        if len(blue_edge_points) > 0:
            avg_x = int(np.mean(blue_edge_points[:, 0]))
            avg_y = int(np.mean(blue_edge_points[:, 1]))
            green_dot_positions_x.append(avg_x)
            green_dot_positions_y.append(avg_y)
            time_points.append(frame_number)  
            cv2.circle(roi, (avg_x, avg_y), 3, (0, 255, 0), -1)
            canvas = np.zeros_like(roi)
            valid_coordinates = [coord for coord in rightmost_positions.values() if coord > 0]
            valid_coordinates.sort()
            contour_mask = np.zeros_like(roi)
            if len(valid_coordinates) >= 2:
                for i in range(len(valid_coordinates) - 1):
                    cv2.line(contour_mask, (valid_coordinates[i], i), (valid_coordinates[i + 1], i + 1), (255, 0, 0), 1)
            stacked_roi = cv2.hconcat([roi, contour_mask])
            cv2_imshow(stacked_roi)
            black_and_blue = cv2.hconcat([canvas, contour_mask])
            cv2_imshow(black_and_blue)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_number += 1
video_capture.release()
cv2.destroyAllWindows()