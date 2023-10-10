# # import cv2
# import numpy as np

# # # Define the color range you want to extract (in this case, green)
# # lower_green = np.array([40, 40, 40])
# # upper_green = np.array([80, 255, 255])

# # # Initialize the video capture object
# # cap = cv2.VideoCapture(r'S:\shasmeen\0.5M sol.MOV')  # Replace with your video file path

# # frame_width = int(cap.get(3))
# # frame_height = int(cap.get(4))

# # # Define the codec and create a VideoWriter object to save the video
# # fourcc = cv2.VideoWriter_fourcc(*'XVID')
# # out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (frame_width, frame_height))

# # while True:
# #     ret, frame = cap.read()

# #     if not ret:
# #         break

# #     # Convert the frame to the HSV color space
# #     hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# #     # Create a mask for the specified color range (green)
# #     mask = cv2.inRange(hsv_frame, lower_green, upper_green)

# #     # Apply the mask to the original frame
# #     result = cv2.bitwise_and(frame, frame, mask=mask)

# #     # Apply thresholding (you can adjust the threshold value)
# #     _, thresholded = cv2.threshold(result, 100, 255, cv2.THRESH_BINARY)

# #     # Display the thresholded image
# #     cv2.imshow('Thresholded Image', thresholded)

# #     # Save the thresholded image
# #     cv2.imwrite('thresholded_image.png', thresholded)

# #     # Write the frame to the output video
# #     out.write(frame)

# #     if cv2.waitKey(10) & 0xFF == ord('q'):
# #         break

# # cap.release()
# # out.release()
# # cv2.destroyAllWindows()



# # open-cv library is installed as cv2 in python
# # import cv2 library into this program
# import cv2
# import matplotlib.pyplot as plt

# # read an image using imread() function of cv2
# # we have to  pass only the path of the image
# image = cv2.imread(r'S:\shasmeen\Screenshot 2023-09-19 114345.png')

# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # Extract the Hue and Value channels from the HSV image
# hue_channel = hsv_image[:, :, 0]
# value_channel = hsv_image[:, :, 2]

# # Create a figure with subplots to display the original image and the Hue and Value channels
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

# # Display the original image
# ax1.set_title('Original Image')
# ax1.imshow(image)

# # Display the Hue channel
# ax2.set_title('Hue Channel')
# ax2.imshow(cv2.cvtColor(hsv_image, cv2.COLOR_BGR2RGB))

# x = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2RGB)


# lower_green = np.array([150, 53, 32])
# upper_green = np.array([168, 80, 35])

# mask = cv2.inRange(hsv_image, lower_green, upper_green)
# result = cv2.bitwise_and(x, x, mask=mask)


# # Display the Value channel
# ax3.set_title('Mask Channel')
# ax3.imshow(result)





# FOR Image
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np

# image = cv2.imread('S:\shasmeen\Screenshot 2023-09-19 114345.png')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.imshow(image)
# plt.show()
# nemo = image
# hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)
# hsv_nemo = image
# light_orange = (149, 155, 100)
# dark_orange = (191, 189, 149)

# mask = cv2.inRange(hsv_nemo, light_orange, dark_orange)
# result = cv2.bitwise_and(nemo, nemo, mask=mask)
# contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # print(contours)
# contour_image = np.zeros_like(nemo)
# largest_contour = max(contours, key=cv2.contourArea)
# # Draw the contours on the blank image
# cv2.drawContours(contour_image, largest_contour, -1, (255, 0, 0), 5)  # Draw in blue

# plt.subplot(1, 2, 1)
# plt.imshow(result)
# # plt.subplot(1, 3, 2)
# # plt.imshow(nemo)
# plt.subplot(1, 2, 2)
# plt.imshow(contour_image)
# plt.show()


#FOR VEDIO

import cv2
import numpy as np

# Initialize the video capture object
cap = cv2.VideoCapture("S:\shasmeen\\0.5M sol.MOV")  # Replace 'your_video.mp4' with your video file path

# Get the video frame dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create a VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 30.0, (frame_width, frame_height))

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

    # Write the frame to the output video
    out.write(frame)

    # Display the processed frame
    cv2.imshow("img", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
