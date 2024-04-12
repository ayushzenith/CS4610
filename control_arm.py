import sys
import time
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import numpy as np
import cv2
import platform
import grid as g

cap = cv2.VideoCapture(0)

def best_move(board):
    if board is None:
        return None  # or handle this case appropriately

    best_score = float('-inf')
    move = None
    for i in range(3):
        for j in range(3):
            if board[i][j] == '':
                board[i][j] = 'O'
                score = minimax(board, 0, False)
                board[i][j] = ''
                # if this score is better than the current best score, update best score and move
                if score > best_score:
                    best_score = score
                    move = (i, j)

    return move # returns the best move for the robot in the form of (row, col)


def minimax(board, depth, is_maximizing):
    winner = check_winner(board)
    if winner == 'O':
        return 1
    elif winner == 'X':
        return -1
    elif winner == None: # tie !
        return 0
    
    if is_maximizing:
        best_score = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == '':
                    board[i][j] = 'O'
                    # move(i, j)
                    score = minimax(board, depth + 1, False)
                    board[i][j] = ''
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == '':
                    board[i][j] = 'X'
                    score = minimax(board, depth + 1, True)
                    board[i][j] = ''
                    best_score = min(score, best_score)
        return best_score


def check_winner(board):
    # check rows
    for row in board:
        if row.count(row[0]) == len(row) and row[0] != '':
            return row[0]

    # check cols
    for col in range(len(board[0])):
        check = []
        for row in board:
            check.append(row[col])
        if check.count(check[0]) == len(check) and check[0] != '':
            return check[0]

    # check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != '':
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != '':
        return board[0][2]

    # check if board is not full / still moves left 
    for row in board:
        for cell in row:
            if cell == '':
                return ''

    # else, tie ! 
    return None



'''
Takes in an i and j (representing the row and column of the board) and converts it to the robot's reference frame
'''
def move(i, j, x, y, dx, dy): 
    '''
    Takes in an i and j (representing the row and column of the board) and moves the robot to that position
    '''
    """
    (-1,-1) (-1,0) (-1,1)
    (0,-1) (0,0) (0,1)
    (1,-1) (1,0) (1,1)
    """
    i = i - 1 # convert from 0 indexed 2d arr to center indexed 2d arr
    j = j - 1

    bot = InterbotixManipulatorXS(        
        robot_model='wx250',
        group_name='arm',
        gripper_name='gripper'
        )
        
    if (bot.arm.group_info.num_joints < 5):
        bot.core.get_logger().fatal('This demo requires the robot to have at least 5 joints!')
        bot.shutdown()
        sys.exit()

    """
    robot frame (in meters) is like
          x
          ^
          |
    y <---
    """
    # the center of the circle the robot will be drawing
    # IN ROBOTS FRAME: equivalent to center square bottom left (x, y) + (dx/2, dy/2) to get the middle of the center
    # then displaced by dx*i, dy*j to get the center of the square to play in 

    start_x, start_y = (x + dx/4) + (dx*-i), (y + dy/4) + (dy*-j)
    print(start_x, start_y)
    ## multiply dx and dy (in reference from the center square) and multiply by some constant 
    # a little bit of clearance so it doesn't initially draw
    RANDOM_UPPER_OFFSET = 0.015
    bot.arm.set_ee_pose_components(x=start_x, y=start_y, z=.0915+RANDOM_UPPER_OFFSET, moving_time=1)
    time.sleep(1)

    center = np.array([start_x, start_y, 0.0915])
    # radius = 0.05
    radius = 0.025
    flag = True
    num_points = 200
    for i in range(num_points+20): # +20 since it doesn't complete the circle at +0
        theta = 2 * np.pi * i / num_points
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        z = center[2]
        if (flag):
            bot.arm.set_ee_pose_components(x=x, y=y, z=z+RANDOM_UPPER_OFFSET, moving_time=1)
            flag = False
        bot.arm.set_ee_pose_components(x=x, y=y, z=z, moving_time=0.05)

    bot.arm.set_ee_pose_components(x=x, y=y, z=z+0.01, moving_time=2)  
    
    
    bot.arm.go_to_home_pose()
    bot.arm.go_to_sleep_pose()
    bot.shutdown()


# returns a lambda
def fit_linear_line(pt1_tuple, pt2_tuple):
    x0, y0 = pt1_tuple
    x1, y1 = pt2_tuple

    slope = (y1 - y0) / (x1 - x0)
    """
    point slope form
    y - y0 = m(x - x0)
    y = m(x - x0) + y0
    """
    return lambda x: slope * (x - x0) + y0


def pixel_space_to_robot_frame(pixel_x, pixel_y):
    """
    image wise (in pixels), it will probably be something like

    (0,0)
    _____________
    |           |
    |           |
    |___________|
       (robot)


    robot frame (in meters) is like
          x
          ^
          |
    y <---

    (sourced from https://docs.trossenrobotics.com/interbotix_xsarms_docs/python_ros_interface.html)
    from the top down
    """
    """
    pixel_x, pixel_y robot_x, robot_y
    (100, 150)       (10, 15)
    (200, 250)       (5, 10)

    pixel_x corresponds to robot_y,
    pixel_y corresponds to robot_x

    aka, find best fit between (150, 10) (250, 5) to get robot_x
    best fit between (100, 15) (200, 10) to get robot_y
    """
    # REPLACE THESE VALUES FOR CALIBRATION
    PT_1_PIXEL_X, PT_1_PIXEL_Y = 442, 171
    PT_1_ROBOT_X, PT_1_ROBOT_Y = 0.25, 0.1

    PT_2_PIXEL_X, PT_2_PIXEL_Y = 141, 362
    PT_2_ROBOT_X, PT_2_ROBOT_Y = 0.4, -0.1

    robot_x_calibration_funct = fit_linear_line((PT_1_PIXEL_Y, PT_1_ROBOT_X),
                                                (PT_2_PIXEL_Y, PT_2_ROBOT_X))

    robot_y_calibration_funct = fit_linear_line((PT_1_PIXEL_X, PT_1_ROBOT_Y),
                                                (PT_2_PIXEL_X, PT_2_ROBOT_Y))

    return robot_x_calibration_funct(pixel_y), robot_y_calibration_funct(pixel_x)

# 400, 235

import grid 

def main():

    gameboard = [['', '', ''],
             ['', '', ''],
             ['', '', '']]


    
    # if platform.system() == 'Windows':
    #     cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    # else:
    # cap = cv2.VideoCapture('./images/testVid3.mp4')
    ret, frame = cap.read()
    if not ret:
        print("Failed to open camera.")
        sys.exit()
    frame = g.detect_board(frame)
    shapeframe = frame.copy()
    edges, contours = g.canny_edge_detection(frame, shapeframe)
    center = g.findCenterRectangle(contours)
    board = [0, 0, frame.shape[1], frame.shape[0]]
    # print('BOARD:', board)
    # print('CENTER:', center)
    grid = [board, center]
    print (center)

    ## bottom left point of center square = (x, y) in pixels = (center[0], center[1])
    ## width, height = (dx, dy) in pixels = (center[2], center[3])
    # center = grid.findCenterRectangle(contours) 

    center_sqr_bottom_left_x, center_sqr_bottom_left_y, center_sqr_bottom_left_width, center_sqr_bottom_left_height = center
    dx, dy = pixel_space_to_robot_frame(center_sqr_bottom_left_width, center_sqr_bottom_left_height)
    center_x, center_y = pixel_space_to_robot_frame(center_sqr_bottom_left_x, center_sqr_bottom_left_y)
    ## CONVERSION FROM PIXEL SPACE TO ROBOTS FRAME WILL JUST HAPPEN ONCE IN THE BEGINGIN 
    print(center)
    print (center_x, center_y, dx, dy)
    
    while True:
        
        ret, frame = cap.read()
        frame = g.detect_board(frame)
        shapeframe = frame.copy()
        cv2.imshow('Tic Tac Toe! Enter m to move', frame)
        # Perform Canny edge detection on the frame
        edges, contours = g.canny_edge_detection(frame, shapeframe)
        if cv2.waitKey(1) & 0xFF == ord('m'):    
            gameboard = g.readBoard(frame, edges, contours, grid, gameboard) # THIS WILL BE THE FUNCTION THAT UPDATES THE BOARD BASED ON THE CV
            print(gameboard)
            move_coords = best_move(gameboard) # Get the best move for the robot
            if move_coords is not None:
                gameboard[move_coords[0]][move_coords[1]] = 'O' # change our internal representation
                print(gameboard)

                ## bottom left = 260
                ## bottom right = 260
                # w, h = 100
                ## pixel to cm = 100 : 10 

            
                # move(move_coords[0],  # this is the row (0, 1, 2)
                #      move_coords[1],  # this is the col (0, 1, 2)
                #      center_x, # this is the bottom left x position of the center square IN ROBOTS FRAME
                #      center_y,  # this is the bottom left y position of the center square IN ROBOTS FRAME
                #      dx,  # this is the displacement the robot needs to move in the x direction to get to the next square IN ROBOTS FRAME 
                #      dy)  # this is the displacement the robot needs to move in the y direction to get to the next square IN ROBOTS FRAME
            else:
                print("No valid moves left for the robot.") 
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

            

        

if __name__ == '__main__':
    main()


