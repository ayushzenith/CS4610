import cv2
import numpy as np

# cap = cv2.VideoCapture('./images/testVid3.mp4')
# cap = cv2.VideoCapture(0)

def canny_edge_detection(frame, shapeframe):
  # Convert the frame to grayscale for edge detection
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # Apply Gaussian blur to reduce noise and smoothen edges
  blurred = cv2.GaussianBlur(src=gray, ksize=(9, 9), sigmaX=0.5)

  # Perform Canny edge detection
  edges = cv2.Canny(blurred, 70, 135)

  # # Turn into thresholded binary
  ret, thresh = cv2.threshold(blurred,125,255, 0)

  # #find and draw contours. RETR_EXTERNAL retrieves only the extreme outer contours
  board, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  # Draw outer contours (corners of the board)
  cv2.drawContours(shapeframe, board, -1, (0,255,0), 3)
  cv2.drawContours(shapeframe, contours, -1, (255,0,0), 3)

  return edges, contours


def get_approx_poly(contour):
  epsilon = 0.04 * cv2.arcLength(contour, True)
  poly_contour = cv2.approxPolyDP(contour, epsilon, True)
  return poly_contour


def detect_board(frame):
  img = frame
  # convert to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # threshold
  thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]

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



def findCenterRectangle(contours):
  rects = []
  for contour in contours:
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)
    if vertices == 4:
      (x, y, w, h) = cv2.boundingRect(approx)
      rects.append([x, y, w, h])

  rects.sort(key=lambda r: r[2] * r[3])
  print (len(rects))
  return rects[-2]


'''
Determine which tic-tac-toe space an (x, y) coordinate is located in.
x:    x coordinate
y:    y coordinate
grid: array [board, center] where each element is [x, y, w, h] for the respective rectangle
Returns a tuple (r, c) where r is the index of the row and c is the index of the column in the tic-tac-toe grid.
'''
def findGridCoordinate(x, y, grid):
  board, center = grid

  b_rows = [board[1], board[1] + board[3]]  # y-coordinates of board top/bottom
  b_cols = [board[0], board[0] + board[2]]  # x-coordinates of board left/right
  rows = [center[1], center[1] + center[3]] # y-coordinates of center top/bottom
  cols = [center[0], center[0] + center[2]] # x-coordinates of center left/right

  # Determine correct row
  if b_rows[0] < y < rows[0]:
    r = 0
  elif rows[0] < y < rows[1]:
    r = 1
  elif rows[1] < y < b_rows[1]:
    r = 2
  else:
    r = -1
  # Determine correct col
  if b_cols[0] < x < cols[0]:
    c = 0
  elif cols[0] < x < cols[1]:
    c = 1
  elif cols[1] < x < b_cols[1]:
    c = 2
  else:
    c = -1

  return (r, c)



def readBoard(frame, edges, contours, grid, gameState):
  '''
  given a 3x3 array of gamestate (chars '', 'O', 'X'), updates it with the current state of the board from camera
  (chars '', 'O', 'X')
  0 = '' = blank
  1 = 'X' = triangle
  2 = 'O' circle
  '''
  # Shape detection on canny edge
  # Convert the frame to grayscale
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # Apply a blur to reduce noise
  gray = cv2.medianBlur(gray, 5)
  rows = gray.shape[0]

  circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,param1=200, param2=35,minRadius=5, maxRadius=75)
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

    if vertices >= 7 and vertices <= 16 and solidity < 0.3 and convex_hull_area > 500 and convex_hull_area < 30000:
      shape = "X"
      M = cv2.moments(contour)
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])
      pos = findGridCoordinate(cX, cY, grid)
      gameState[pos[0]][pos[1]] = 'X'
    else:
      shape = ""

    cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
    cv2.putText(frame, shape, (approx[0][0][0], approx[0][0][1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    # print(approx)
  
  # If circles are detected
  if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
      # Draw the outer circle
      cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
      # Draw the center of the circle
      cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
      pos = findGridCoordinate(i[0], i[1], grid)
      cv2.putText(frame, f"O {pos}", (i[0], i[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0), 2)
      gameState[pos[0]][pos[1]] = 'O'

  return gameState
  # Display the frame with the drawn circles
  cv2.imshow('detected circles', frame)








# # Get center rectangle, outer board rectangle
# ret, frame = cap.read()
# frame = detect_board(frame)
# shapeframe = frame.copy()
# edges, contours = canny_edge_detection(frame, shapeframe)
# center = findCenterRectangle(contours)
# board = [0, 0, frame.shape[1], frame.shape[0]]
# print('BOARD:', board)
# print('CENTER:', center)
# grid = [board, center]
# print (center)


# while True:
#   ret, frame = cap.read()

#   frame = detect_board(frame)
#   shapeframe = frame.copy()
#   cv2.imshow('frame', frame)
#   # Perform Canny edge detection on the frame
#   edges, contours = canny_edge_detection(frame, shapeframe)

#   # Display center box
#   # center = findCenterRectangle(contours)
#   cv2.rectangle(shapeframe, [center[0], center[1]], [center[0] + center[2], center[1] + center[3]], color=(0,0,255), thickness=3)
#   cv2.rectangle(shapeframe, [0, 0], [frame.shape[1], frame.shape[0]], color=(0,0,255), thickness=3)
#   cv2.circle(shapeframe, [center[0], center[1]], 5, color=(0,255,0), thickness=5)

#   # Display the original frame and the edge-detected frame
#   cv2.imshow("Original", frame)
#   cv2.imshow("Edges", edges)
#   cv2.imshow("Grid", shapeframe)

#   if cv2.waitKey(33) == ord('a'):
#     ## modified this to chars, to work with the control_arm impl! 
#     gameState = [['', '', ''],
#            ['', '', ''],
#            ['', '', '']]
#     readBoard(frame, edges, contours, grid,gameState)
#     for row in gameState:
#       print(row)

#   if cv2.waitKey(1) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()
#     break
