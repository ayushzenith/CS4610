import cv2
import numpy as np

# Open the video file
cap = cv2.VideoCapture('images/testVid3.mp4')

def detect_board(frame):
    img = frame
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold
    thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1] \
 \
        # apply morphology
    kernel = np.ones((7, 7), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((9, 9), np.uint8)
    morph = cv2.morphologyEx(morph, cv2.MORPH_ERODE, kernel)

    # get largest contour
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    area_thresh = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > area_thresh:
            area_thresh = area
            big_contour = c

    # get bounding box
    x, y, w, h = cv2.boundingRect(big_contour)

    # draw filled contour on black background
    mask = np.zeros_like(gray)
    mask = cv2.merge([mask, mask, mask])
    cv2.drawContours(mask, [big_contour], -1, (255, 255, 255), cv2.FILLED)

    # apply mask to input
    result1 = img.copy()
    result1 = cv2.bitwise_and(result1, mask)

    # crop result
    result2 = result1[y:y + h, x:x + w]

    return result2


def canny_edge_detection(frame):
    # Convert the frame to grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and smoothen edges
    blurred = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0.5)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 70, 135)

    return blurred, edges

def get_approx_poly(contour):
    epsilon = 0.04 * cv2.arcLength(contour, True)
    poly_contour = cv2.approxPolyDP(contour, epsilon, True)
    return poly_contour


while(cap.isOpened()):
    # Read the video frame by frame
    ret, frame = cap.read()
    if ret:

        frame = detect_board(frame)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply a blur to reduce noise
        gray = cv2.medianBlur(gray, 5)

        rows = gray.shape[0]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,param1=100, param2=30,minRadius=1, maxRadius=100)
        # Detect circles using HoughCircles
        #circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=1, maxRadius=100)

        blurred, edges = canny_edge_detection(frame)

        # Shape detection on canny edge
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        for contour in contours:
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            vertices = len(approx)

            contour_area = cv2.contourArea(contour)
            poly = get_approx_poly(contour)
            convex = cv2.isContourConvex(poly)

            convex_hull = cv2.convexHull(contour)
            convex_hull_area = cv2.contourArea(convex_hull)
            try:
                solidity = contour_area / convex_hull_area
            except ZeroDivisionError:
                solidity = 0

            if vertices >= 8 and vertices <= 10 and solidity < 0.5 and convex_hull_area > 1000 and convex_hull_area < 10000:
                #shape = "X" + str(vertices) + " " + str(convex_hull_area) + " " + str(solidity)
                shape = "X" + str(convex)
            else:
                shape = ""

            #cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
            cv2.putText(frame, shape, (approx[0][0][0], approx[0][0][1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # If circles are detected
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw the outer circle
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
                cv2.putText(frame, "O", (i[0], i[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0), 2)

        # Display the frame with the drawn circles
        cv2.imshow('detected circles', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
