import cv2
import platform
import numpy as np
import time

GREEN = (0, 255, 0)

# Capture video from webcam if you want to use default webcam just pass 0 as argument instead of 1
if platform.system() == 'Windows':
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(1)


def canny_edge_detection(frame):
    # Convert the frame to grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and smoothen edges
    blurred = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0.5)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 70, 135)

    return blurred, edges


while True:
    # time.sleep(1)
    # Read a frame from the webcam
    ret, frame = cap.read()
    shapeframe = frame.copy()
    if not ret:
        print('Image not captured')
        break

    # Perform Canny edge detection on the frame
    blurred, edges = canny_edge_detection(frame)

    # Shape detection on canny edge
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True) # approx is a numpy array
        vertices = len(approx)

        if vertices == 3:
            shape = "Triangle"
        elif vertices == 4:
            rect = cv2.minAreaRect(approx)
            # cv.boxPoints(rect) to get 4 corners of rect
            # Angle between the horizontal axis and the first side (i.e. length) in degrees
            (x, y), (w, h), angle = rect
            ar = w / float(h)
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        else:
            shape = "Circle"

        cv2.drawContours(shapeframe, [approx], 0, GREEN, 2)
        cv2.putText(shapeframe, shape, (approx[0][0][0], approx[0][0][1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    into_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # changing the color format from BGr to HSV
    # This will be used to create the mask
    L_limit = np.array([0, 0, 0])  # setting the black lower limit
    U_limit = np.array([50, 50, 150])  # setting the black upper limit

    b_mask = cv2.inRange(into_hsv, L_limit, U_limit)
    # creating the mask using inRange() function
    # this will produce an image where the color of the objects
    # falling in the range will turn white and rest will be black
    black = cv2.bitwise_and(frame, frame, mask=b_mask)


    # Display the original frame and the edge-detected frame
    cv2.imshow("Original", frame)
    cv2.imshow("Blurred", blurred)
    cv2.imshow("Edges", edges)
    cv2.imshow("Shapes", shapeframe)
    cv2.imshow('Black Detector', black)  # to display the black object output


    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()

# recognize a handdrawn tictactoe board in opencv and python and write the code to only show board and moves made and keep track of the moves




"""
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
"""
"""
import cv2
import numpy as np

def nothing(x):
    # any operation
    pass
cap = cv2.VideoCapture(1)
cv2.namedWindow("Trackbars")
"""
"""
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("result", result)

    key = cv2.waitKey(1)
    if key == 27:
        break
"""
