import cv2
import numpy as np
import matplotlib.pyplot as plt
import platform
import grid

'''
Images:
  - full-clean-board.png  : default stock image of board (no tilt)
  - tilted-board.png      : slightly tilted stock image of board
  - board1-crop.jpg       : real image from camera with reflection (cropped to remove tags)
'''

# frame = cv2.imread('images/board1-crop.jpg')
# shapeframe = frame.copy()

if platform.system() == 'Windows':
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(0)



def canny_edge_detection(frame):
  # Convert the frame to grayscale for edge detection
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # Apply Gaussian blur to reduce noise and smoothen edges
  blurred = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0.5)

  # Perform Canny edge detection
  edges = cv2.Canny(blurred, 70, 135)

  # # Turn into thresholded binary
  ret, thresh = cv2.threshold(blurred,110,255, 0)

  # #find and draw contours. RETR_EXTERNAL retrieves only the extreme outer contours
  board, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  # Draw outer contours (corners of the board)
  # cntrs = blurred.copy()
  cv2.drawContours(shapeframe, board, -1, (0,255,0), 3)
  cv2.drawContours(shapeframe, contours, -1, (255,0,0), 3)

  return edges, contours


def printEdges(x, y, points):
  plt.scatter(x, y)
  for p in points:
    plt.scatter(p[0], p[1], marker='x', color='red')
  plt.show()

def findSquares(contours):
  rects = []
  for contour in contours:
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)
    if vertices == 4:
      (x, y, w, h) = cv2.boundingRect(approx)
      rects.append([x, y, w, h])
  # print(rects)
  return rects # returns board rectangle, ceneter rectangle (use rects[1])

def findGrid(board, center):
  # board: 4 element array, x, y (of bottom left corner of board), width, height
  # center: 4 element array, x, y, width, height
  # returns 3x3 array

  x = board[1]
  y = board[0]
  b1 = [x, y, center[1] - x, center[0] - y]
  x += b1[2]
  b2 = [x, y, center[3], center[0] - y]
  x += b2[2]
  b3 = [x, y, board[1] + board[3] - x, center[0] - y]

  x = board[1]
  y = center[0]
  m1 = [x, y, center[1] - x, center[2]]
  x += m1[2]
  m2 = [center[1], center[0], center[3], center[2]]
  x += m2[2]
  m3 = [x, y, board[1] + board[3] - x, center[2]]

  x = board[1]
  y += center[2]
  t1 = [x, y, center[1] - x, board[0] + board[2] - y]
  x += t1[2]
  t2 = [x, y, center[3], board[0] + board[2] - y]
  x += t2[2]
  t3 = [x, y, board[1] + board[3] - x, board[0] + board[2] - y]

  grid = [[t1, t2, t3],
          [m1, m2, m3],
          [b1, b2, b3]]
  corners = [
    [[], [], []],
    [[], [], []],
    [[], [], []],
  ]
  for i, r in enumerate(grid):
    for j, sq in enumerate(r):
      corners[i][j] = [(sq[0], sq[1]), (sq[0] + sq[2], sq[1] + sq[3])]
  return grid, corners


def drawGrid(frame, grid):
  for i in range(len(grid)):
    for j, tile in enumerate(grid[i]):
      cv2.rectangle(frame, [tile[1], tile[0]], [tile[1] + tile[3], tile[0] + tile[2]], color=(0,0,255), thickness=3)


while True:
  # Perform Canny edge detection on the frame
  ret, frame = cap.read()
  shapeframe = frame.copy()
  edges, contours = canny_edge_detection(frame)

  # b, c = findSquares(contours)
  # grid, corners = findGrid(b, c)
  # drawGrid(shapeframe, grid)

  # # Display the original frame and the edge-detected frame
  # cv2.imshow("Original", frame)
  # cv2.imshow("Edges", edges)
  # cv2.imshow("Grid", shapeframe)

  # indices = np.where(edges != [0])
  # y_coords = indices[1]
  # x_coords = indices[0]

  if cv2.waitKey(1) & 0xFF == ord('q'):
    # w, h, p = boardStats(board[0])
    ##printEdges(x_coords, y_coords, points=[])
    # print('Image top left:', (0, frame.shape[1]))

    board, c = findSquares(contours)
    print(board, c)
    grid, corners = findGrid(board, c)
    print(grid)
    print(corners)
    cv2.destroyAllWindows()
    break
cap.release()