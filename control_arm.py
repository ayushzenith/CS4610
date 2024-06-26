import sys
import time
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import numpy as np
import cv2
import platform
import grid as g

cap = cv2.VideoCapture(0)

import numpy as np


def best_move(board):
    if board is None:
        return None  # or handle this case appropriately

    best_score = float('-inf')
    move = None
    for i in range(2, -1,-1):
        for j in range(3):
            if board[i][j] == '':
                board[i][j] = 'X'
                score = minimax(board, 0, True)
                board[i][j] = ''
                # if this score is better than the current best score, update best score and move
                if score > best_score:
                    best_score = score
                    move = (i, j)

    return move # returns the best move for the robot in the form of (row, col)

def minimax(board, depth, is_maximizing):
    winner = check_winner(board)
    if winner == 'X':
        return 10 - depth
    elif winner == 'O':
        return depth - 10
    elif winner is None:  # Tie
        return 0

    if is_maximizing:  # 'X'
        best_score = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == '':
                    board[i][j] = 'X'
                    score = minimax(board, depth + 1, False)  # Switch to minimize for 'O'
                    board[i][j] = ''
                    best_score = max(score, best_score)
        return best_score
    else:  # 'O'
        best_score = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == '':
                    board[i][j] = 'O'
                    score = minimax(board, depth + 1, True)  # Switch to maximize for 'X'
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
    print(f"Move called with i={i}, j={j}, x={x}, y={y}, dx={dx}, dy={dy}")
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
    # IN ROBOTS FRAME: equivalent to center square top left (x, y) - (dx/2, dy/2) to get the middle of the center
    # then displaced by dx*i, dy*j to get the center of the square to play in 
    # both displacements are negatived due to how the robot frame coordinates increase left and top
    start_x, start_y = (x - dx/2) + (dx*-i), (y - dy/4) + (dy*-j) # TOOD div by 2
    print(f"start_x={start_x}, start_y={start_y}")

    # a little bit of clearance so it doesn't initially draw
    RANDOM_UPPER_OFFSET = 0.025
    bot.arm.set_ee_pose_components(x=start_x, y=start_y, z=.0915+RANDOM_UPPER_OFFSET, moving_time=1)
    time.sleep(1)

    center = np.array([start_x, start_y, 0.1])
    # radius = 0.05
    radius = 0.02
    flag = True

    bot.arm.set_ee_pose_components(x=start_x, y = start_y,  z=0.2, moving_time=2)
    bot.arm.set_ee_pose_components(x=start_x, y = start_y,  z=0.09, moving_time=2)
    bot.arm.set_ee_pose_components(x=start_x, y = start_y,  z=0.2, moving_time=2)


    # for i in range(num_points+20): # +20 since it doesn't complete the circle at +0
    #     theta = 2 * np.pi * i / num_points
    #     x = center[0] + radius * np.cos(theta)
    #     y = center[1] + radius * np.sin(theta)
    #     z = center[2]
    #     if (flag):
    #         bot.arm.set_ee_pose_components(x=x, y=y, z=z+RANDOM_UPPER_OFFSET, moving_time=1)
    #         flag = False
    #     bot.arm.set_ee_pose_components(x=x, y=y, z=z, moving_time=0.05)
    
    
    bot.arm.go_to_home_pose()
    bot.arm.go_to_sleep_pose()
    bot.shutdown()


    
    # center = np.array([start_x, start_y, 0.1])
    # # radius = 0.05
    # radius = 0.025
    # flag = True
    # num_points = 300
    # for i in range(num_points+20): # +20 since it doesn't complete the circle at +0
    #     theta = 2 * np.pi * i / num_points
    #     x = center[0] + radius * np.cos(theta)
    #     y = center[1] + radius * np.sin(theta)
        
    #     # if np.isclose(theta, np.pi):
    #     #     z = center[2] - 0.01
    #     # else:
    #     z = center[2] - 0.02
    #     if (flag):
    #         bot.arm.set_ee_pose_components(x=x, y=y, z=z+RANDOM_UPPER_OFFSET, moving_time=1)
    #         flag = False
    #     if (bot.arm.set_ee_pose_components(x=x, y=y, z=z, moving_time=0.05) == False):
    #         print("Failed to move to position")
    #         break
    # time.sleep(1)
    # if (bot.arm.set_ee_pose_components(x=x, y=y, z=z+0.1, moving_time=2) == False):
    #         print("Failed to move to position")
                
    
    # bot.arm.go_to_home_pose()
    # bot.arm.go_to_sleep_pose()
    # bot.shutdown()


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
    PT_1_PIXEL_X, PT_1_PIXEL_Y = 427, 123
    PT_1_ROBOT_X, PT_1_ROBOT_Y = .5, -.2

    PT_2_PIXEL_X, PT_2_PIXEL_Y = 87, 326
    PT_2_ROBOT_X, PT_2_ROBOT_Y = .3, .2

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
    print(f"Raw center coordinates from img: {center}")

    ## bottom left point of center square = (x, y) in pixels = (center[0], center[1])
    ## width, height = (dx, dy) in pixels = (center[2], center[3])
    # center = grid.findCenterRectangle(contours) 

    center_sqr_top_left_x, center_sqr_top_left_y, center_sqr_top_left_width, center_sqr_top_left_height = center

    bottom_right_x, bottom_right_y = pixel_space_to_robot_frame(
        center_sqr_top_left_x + center_sqr_top_left_width,
        center_sqr_top_left_y + center_sqr_top_left_height)

    center_x, center_y = pixel_space_to_robot_frame(center_sqr_top_left_x, center_sqr_top_left_y)

    dx, dy = (center_x - bottom_right_x), (center_y - bottom_right_y)
    ## CONVERSION FROM PIXEL SPACE TO ROBOTS FRAME WILL JUST HAPPEN ONCE IN THE BEGINGIN 
    print(f"Robot coordinates: {center_x}, {center_y}, {dx}, {dy}")

    while True:
        
        ret, frame = cap.read()
        # frame = g.detect_board(frame)
        shapeframe = frame.copy()
        cv2.imshow('Tic Tac Toe! Enter m to move', frame)
        # Perform Canny edge detection on the frame
        edges, contours = g.canny_edge_detection(frame, shapeframe)

        pressed_key = cv2.waitKey(1)
        # s is for scanning and updating board state, m is for actually moving
        # the idea is that we can scan multiple times now, to redraw shapes if necessary
        # before moving
        if pressed_key & 0xFF == ord('s'):
            gameboard = g.readBoard(frame, edges, contours, grid, gameboard) # THIS WILL BE THE FUNCTION THAT UPDATES THE BOARD BASED ON THE CV
            print ("befor robot move")
            for row in gameboard:
                print(row)
        elif pressed_key & 0xFF == ord('m'):
            winner = check_winner(gameboard)
            if winner != "":
                print("Game ended!")
                if winner is None:
                    print("Tie!")
                else:
                    print(f"Winner is {winner}")

                cv2.destroyAllWindows()
                break

            move_coords = best_move(gameboard) # Get the best move for the robot
            if move_coords is not None:
                gameboard[move_coords[0]][move_coords[1]] = 'X' 
                print ("after robot move")
                for row in gameboard:
                    print(row)
            
                move(move_coords[0],  # this is the row (0, 1, 2)
                     move_coords[1],  # this is the col (0, 1, 2)
                     center_x, # this is the bottom left x position of the center square IN ROBOTS FRAME
                     center_y,  # this is the bottom left y position of the center square IN ROBOTS FRAME
                     dx,  # this is the displacement the robot needs to move in the x direction to get to the next square IN ROBOTS FRAME 
                     dy)  # this is the displacement the robot needs to move in the y direction to get to the next square IN ROBOTS FRAME
            else:
                print("No valid moves left for the robot.") 
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

            

        

if __name__ == '__main__':
    main()


