import cv2
import numpy as np
import matplotlib.pyplot as plt

# Initialize the video capture object
cap = cv2.VideoCapture("S:\shasmeen\\0.5M sol.MOV")

# Get the video frame dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create a VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 30.0, (frame_width, frame_height))

# Define the fixed y-coordinate
fixed_y = 300  # You can adjust this to the desired y-coordinate

# Lists to store the x-coordinate and time values
x_values = []
time_values = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    light_orange = (149, 155, 100)
    dark_orange = (191, 189, 149)

    mask = cv2.inRange(frame, light_orange, dark_orange)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour by area
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rightmost_point = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])

        # Draw only the largest contour in blue
        cv2.drawContours(frame, [largest_contour], -1, (0, 0, 255), 5)

        # Draw the rightmost point as a red circle
        cv2.circle(frame, rightmost_point, 10, (255, 0, 0), -1)

        # Append x-coordinate and current time to the lists
        x_values.append(rightmost_point[0])
        time_values.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)  # Convert time to seconds

    # Write the frame to the output video
    out.write(frame)

    # Display the processed frame
    cv2.imshow("img", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Plot the x-coordinate variation over time
plt.plot(time_values, x_values)
plt.xlabel("Time (s)")
plt.ylabel("X-coordinate")
plt.title("X-coordinate Variation Over Time")
plt.grid()
plt.show()
